"""Registered cross-query transfer through selected MLP neuron coordinates."""

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_backend import CausalWorkspace, qualify_cached
from .causal_campaign import condition_seed, meets, rates
from .tasks import build_tasks


LAYERS = (7, 15, 23, 31)
COUNTS = (16, 64, 256)
ALTERNATE = {"capital": "currency", "currency": "capital_continent", "capital_continent": "language", "language": "capital"}
CAP_SECONDS = 2700


def contribution_ranking(changes, column_norms, relations):
    """Rank by minimum relation-mean share of rescaling-invariant write energy."""
    changes, column_norms = np.asarray(changes, dtype=np.float64), np.asarray(column_norms, dtype=np.float64)
    if changes.ndim != 2 or changes.shape[1:] != column_norms.shape or len(changes) != len(relations):
        raise ValueError("Neuron ranking dimensions mismatch")
    if not np.isfinite(changes).all() or not np.isfinite(column_norms).all():
        raise ValueError("Nonfinite neuron data")
    energies = changes**2 * column_norms[None]**2
    totals = energies.sum(axis=1, keepdims=True)
    shares = np.divide(energies, totals, out=np.zeros_like(energies), where=totals > 0)
    groups = {r: shares[np.array(relations) == r].mean(axis=0) for r in sorted(set(relations))}
    if set(groups) != set(ALTERNATE):
        raise ValueError("All four relations required for neuron ranking")
    score = np.stack(list(groups.values())).min(axis=0)
    order = np.argsort(-score, kind="stable")
    return order, score, groups


def selected_increment(change, units):
    change = np.asarray(change, dtype=np.float32)
    units = np.asarray(units, dtype=int)
    if len(set(units.tolist())) != len(units) or (units < 0).any() or (units >= len(change)).any():
        raise ValueError("Invalid neuron selection")
    result = np.zeros_like(change)
    result[units] = change[units]
    return result


def project_neurons(model, layer, increment):
    mx = model.mx
    vector = model.layers[layer].mlp.down_proj.weight @ mx.array(increment, dtype=mx.float32)
    mx.eval(vector)
    result = np.array(vector)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite neuron write")
    return result


def random_neuron_edit(model, layer, count, excluded, reference, seed):
    width = model.layers[layer].mlp.down_proj.weight.shape[1]
    rng = np.random.default_rng(seed)
    allowed = np.setdiff1d(np.arange(width), excluded)
    if count > len(allowed):
        raise ValueError("Not enough random control neurons")
    units = rng.choice(allowed, size=count, replace=False)
    increment = np.zeros(width, dtype=np.float32)
    increment[units] = rng.normal(size=count)
    delta = project_neurons(model, layer, increment)
    norm = np.linalg.norm(delta)
    target = np.linalg.norm(reference)
    if norm <= 1e-12:
        raise ValueError("Degenerate random neuron write")
    scale = target / norm
    return delta * scale, increment * scale, units


def select_tasks(tasks):
    training = [r for r in tasks if r['family'] != 'surface_copy' and r['paraphrase_id'] == 0 and r['id'].endswith('/d0')]
    evaluation = [r for r in tasks if r['family'] == 'surface_copy' or r['paraphrase_id'] == 1]
    if len(training) != 24 or len(evaluation) != 72:
        raise ValueError("Frozen neuron task counts changed")
    train_prompts = {r['prompt'] for r in training} | {r['donor_prompt'] for r in training}
    if train_prompts & {r['prompt'] for r in evaluation}:
        raise ValueError("Neuron selection/evaluation prompt overlap")
    return training, evaluation


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['layer'], row['count'], row['condition'])].append(row)
    summary = {}
    for (layer, count, condition), members in groups.items():
        item = rates(members)
        item['wrong_query_intrusions'] = sum(r['wrong_query_answer'] for r in members if r['expected_effect'] == 'change')
        for field in ('relation', 'family', 'pair_id'):
            item['by_' + field] = {key: rates([r for r in members if r[field] == key]) for key in sorted({r[field] for r in members})}
        summary.setdefault(str(layer), {}).setdefault(str(count), {})[condition] = item
    return summary


def decision(summary):
    primary = summary['23']['64']
    cross, random = primary['cross_query'], primary['random_neurons']
    rate = cross['counterfactual']['rate']
    control = random['counterfactual']['rate']
    advantage = rate-control if rate is not None and control is not None else None
    relations = {k: v['counterfactual']['rate'] for k, v in cross['by_relation'].items()}
    checks = {'cross_query': meets(rate, .5),
              'random_advantage': meets(advantage, .2),
              'copy': meets(cross['copy_preserved']['rate'], .95),
              'relations': set(relations) == set(ALTERNATE) and all(meets(v, .3) for v in relations.values())}
    return {'supported': all(checks.values()), 'checks': checks, 'cross_query_rate': rate,
            'random_advantage': advantage, 'relations': relations, 'primary_layer': 23, 'primary_neurons': 64,
            'claim_scope': 'Cross-query MLP-coordinate transfer on a paraphrase holdout of known facts; not a complete circuit or decipherment'}


def run():
    config = load_config('configs/jspace0001.json')
    output, result = Path('outputs/NEURON-0001'), Path('results/NEURON-0001')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result / 'results.json').exists():
        raise FileExistsError('Neuron evaluation already completed')
    progress_path = output / 'progress.json'
    prior = json.loads(progress_path.read_text())['seconds'] if progress_path.exists() else 0.
    started = time.monotonic()

    def checkpoint(phase):
        elapsed = prior + time.monotonic()-started
        write_json(progress_path, {'seconds': elapsed, 'phase': phase})
        if elapsed >= CAP_SECONDS:
            write_json(result / 'incomplete.json', {'seconds': elapsed, 'reason': 'time_cap', 'phase': phase})
            raise TimeoutError('Neuron campaign time cap')

    model = CausalWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])
    manifest_path = result / 'inputs.json'
    if not manifest_path.exists():
        qualification = qualify_cached(model)
        write_json(result / 'qualification.json', qualification)
        if not qualification['passed']:
            raise ValueError('Neuron trace/cache qualification failed')
        dependencies = ('__init__.py', 'campaign.py', 'causal_backend.py', 'causal_campaign.py', 'geometry.py', 'mlx_backend.py', 'tasks.py', 'neuron_campaign.py')
        paths = [Path('src/voynich/workspace') / name for name in dependencies] + [Path('docs/experiments/NEURON-0001.md')]
        write_json(manifest_path, {'source_sha256': {str(p): digest(p) for p in paths},
                                  'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                  'jspace_inputs_sha256': digest('results/JSPACE-0001/inputs.json'),
                                  'config_sha256': canonical_digest(config), 'qualification_sha256': digest(result / 'qualification.json'),
                                  'layers': LAYERS, 'counts': COUNTS, 'primary': [23, 64], 'cap_seconds': CAP_SECONDS,
                                  'precision': 'MLX_ENABLE_TF32=0; cached quantized values expanded to float32 in all transformer blocks'})
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Neuron source drift: {path}')
    if canonical_digest(config) != manifest['config_sha256'] or digest('results/JSPACE-0001/inputs.json') != manifest['jspace_inputs_sha256']:
        raise ValueError('Neuron input drift')
    manifest_hash = digest(manifest_path)
    tasks = build_tasks(seed=config['seed'], split='development')
    training, evaluation = select_tasks(tasks)
    trace_dir = output / 'traces'
    trace_dir.mkdir(exist_ok=True)
    memory = {}

    def trace(prompt):
        key = canonical_digest(prompt)
        if key in memory:
            return memory[key]
        checkpoint('trace')
        meta_path, array_path = trace_dir / f'{key}.json', trace_dir / f'{key}.npz'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['inputs_sha256'] != manifest_hash or digest(array_path) != meta['arrays_sha256']:
                raise ValueError('Neuron trace provenance mismatch')
            with np.load(array_path) as data:
                arrays = {k: data[k] for k in data.files}
        else:
            ids = model.encode(prompt, chat=True)
            text, generated, _ = model.greedy_cached(ids, max_tokens=config['max_generated_tokens'])
            _, arrays = model.neuron_trace(ids)
            np.savez(array_path, **arrays)
            meta = {'ids': ids, 'text': text, 'generated_tokens': generated, 'arrays_sha256': digest(array_path), 'inputs_sha256': manifest_hash}
            write_json(meta_path, meta)
        memory[key] = (meta, arrays)
        return meta, arrays

    column_norms = {}
    for layer in LAYERS:
        value = model.mx.sqrt(model.mx.sum(model.layers[layer].mlp.down_proj.weight**2, axis=0))
        model.mx.eval(value)
        column_norms[layer] = np.array(value)
    ranks, selections = {}, {}
    for layer in LAYERS:
        changes = [trace(row['donor_prompt'])[1]['neurons'][layer] - trace(row['prompt'])[1]['neurons'][layer] for row in training]
        order, score, relation_scores = contribution_ranking(changes, column_norms[layer], [r['relation'] for r in training])
        ranks[layer] = order
        selections[str(layer)] = {'top_256': order[:256].tolist(), 'top_scores': score[order[:256]].tolist(),
                                   'nonzero_scores': int((score > 0).sum()),
                                   'top_relation_shares': {k: v[order[:256]].tolist() for k, v in relation_scores.items()}}
    selection = {'inputs_sha256': manifest_hash, 'training_ids': [r['id'] for r in training], 'layers': selections}
    selection_path = result / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Neuron selection drift')
    write_json(selection_path, selection)

    # Independently check that activation-coordinate edits equal residual writes.
    qualification = []
    sample = trace(training[0]['prompt'])[1]
    for layer in LAYERS:
        a = sample['neurons'][layer]
        eta = np.zeros_like(a)
        eta[ranks[layer][:16]] = .03
        linear = model.layers[layer].mlp.down_proj
        explicit = linear(model.mx.array(a+eta)) - linear(model.mx.array(a))
        model.mx.eval(explicit)
        projected = project_neurons(model, layer, eta)
        relative = float(np.linalg.norm(np.array(explicit)-projected) / max(np.linalg.norm(projected), 1e-12))
        qualification.append({'layer': layer, 'relative_error': relative})
    write_json(result / 'neuron-edit-qualification.json', {'passed': all(r['relative_error'] < .002 for r in qualification), 'rows': qualification})
    if not all(r['relative_error'] < .002 for r in qualification):
        raise ValueError('Neuron-to-residual equivalence failed')

    lookup = {(r['family'], r['pair_id'], r['paraphrase_id'], r['relation'], r['latent_concept']): r for r in tasks}
    log_path = output / 'observations.jsonl'
    rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    done = {(r['id'], r['layer'], r['count'], r['condition']) for r in rows}
    if len(done) != len(rows):
        raise ValueError('Duplicate neuron observations')
    with log_path.open('a') as handle:
        for task in evaluation:
            own, own_trace = trace(task['prompt'])
            donor, donor_trace = trace(task['donor_prompt'])
            alternate_family = 'indirect_fact' if task['family'] == 'surface_copy' else task['family']
            alternate_relation = 'capital' if task['family'] == 'surface_copy' else ALTERNATE[task['relation']]
            alternate = lookup[(alternate_family, task['pair_id'], 1, alternate_relation, task['latent_concept'])]
            cross_source = trace(alternate['prompt'])[1]
            cross_donor = trace(alternate['donor_prompt'])[1]
            for layer in LAYERS:
                same = donor_trace['neurons'][layer]-own_trace['neurons'][layer]
                cross = cross_donor['neurons'][layer]-cross_source['neurons'][layer]
                conditions = [(0, 'identity', np.zeros(model.width), []),
                              (len(cross), 'full_mlp_cross_query', project_neurons(model, layer, cross), [])]
                for count in COUNTS:
                    units = ranks[layer][:count]
                    cross_delta = project_neurons(model, layer, selected_increment(cross, units))
                    random_delta, _, random_units = random_neuron_edit(model, layer, count, units, cross_delta,
                                                                      condition_seed(config['seed'], task['id'], layer, f'neurons-{count}'))
                    conditions.extend([(count, 'same_query', project_neurons(model, layer, selected_increment(same, units)), units.tolist()),
                                       (count, 'cross_query', cross_delta, units.tolist()),
                                       (count, 'random_neurons', random_delta, random_units.tolist())])
                for count, condition, delta, units in conditions:
                    if (task['id'], layer, count, condition) in done:
                        continue
                    checkpoint('evaluation')
                    text, generated, _ = model.greedy_cached(own['ids'], patches=[{'layer': layer, 'position': len(own['ids'])-1, 'delta': delta}], max_tokens=config['max_generated_tokens'])
                    if condition == 'identity' and generated != own['generated_tokens']:
                        raise RuntimeError('Neuron identity changed generation')
                    row = {k: task[k] for k in ('id', 'family', 'relation', 'pair_id', 'prompt_group', 'expected_effect', 'latent_concept', 'donor_concept')}
                    row.update({'layer': layer, 'count': count, 'condition': condition, 'answer': text, 'generated_tokens': generated,
                                'valid': True, 'baseline': own['text'], 'donor': donor['text'], 'alternate_relation': alternate_relation,
                                'baseline_correct': answer_correct(own['text'], task['answers']),
                                'donor_correct': answer_correct(donor['text'], task['counterfactual_answers']),
                                'original_correct': answer_correct(text, task['answers']),
                                'counterfactual_correct': answer_correct(text, task['counterfactual_answers']),
                                'wrong_query_answer': task['expected_effect'] == 'change' and answer_correct(text, alternate['counterfactual_answers']),
                                'residual_delta_norm': float(np.linalg.norm(delta)), 'recipient_norm': float(np.linalg.norm(own_trace['residual'][layer])),
                                'neuron_ids': units, 'truncated': len(generated) == config['max_generated_tokens']})
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush()
                    rows.append(row)
                    done.add((task['id'], layer, count, condition))
                    if model.mx.get_peak_memory() > config['memory_bytes_cap']:
                        raise MemoryError('Neuron memory cap exceeded')
                    if len(rows) % 120 == 0:
                        print(json.dumps({'phase': 'neurons', 'completed': len(rows), 'seconds': prior+time.monotonic()-started}), flush=True)
    expected = len(evaluation)*len(LAYERS)*(2+3*len(COUNTS))
    if len(rows) != expected:
        raise ValueError('Incomplete neuron observation grid')
    summary = summary_rows(rows)
    report = {'summary': summary, 'observations': len(rows), 'unique_traces': len(memory),
              'seconds': prior+time.monotonic()-started, 'inputs_sha256': manifest_hash,
              'observations_sha256': digest(log_path), 'selection_sha256': digest(selection_path),
              'peak_memory_bytes': model.mx.get_peak_memory()}
    write_json(result / 'results.json', report)
    result_decision = decision(summary)
    result_decision['results_sha256'] = digest(result / 'results.json')
    write_json(result / 'decision.json', result_decision)
    checkpoint('complete')
    print(json.dumps(result_decision), flush=True)


if __name__ == '__main__':
    run()
