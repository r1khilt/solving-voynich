"""Read-only empirical replay from frozen saved models, plus separate gate equations."""
import argparse
import json
import math
import signal
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import torch

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import (
    CHECKPOINTS, OUT, PATHS, ROOT, SEEDS, TAUS, artifact, limit_resources, load_inputs,
)
from voynich.recurrent_latin_source import ALPHABET, RecurrentSource, chunks, learning_rate, score_records


def verify(spec):
    if artifact(ROOT / spec['path']) != spec:
        raise ValueError('Artifact checksum or byte count mismatch')


@torch.inference_mode()
def manual_logits(model, tokens):
    """Explicit i/f/g/o equations, no nn.LSTM call; one sequence only."""
    params = model.state_dict()
    width, layers = model.config['width'], model.config['layers']
    h = [torch.zeros(width, dtype=torch.float64) for _ in range(layers)]
    c = [torch.zeros(width, dtype=torch.float64) for _ in range(layers)]
    p = {k: v.double() for k, v in params.items()}
    outputs = []
    for token in tokens:
        value = p['embedding.weight'][int(token)]
        for layer in range(layers):
            gates = (p[f'recurrent.weight_ih_l{layer}'] @ value + p[f'recurrent.bias_ih_l{layer}']
                     + p[f'recurrent.weight_hh_l{layer}'] @ h[layer] + p[f'recurrent.bias_hh_l{layer}'])
            i, f, g, o = gates.chunk(4)
            c[layer] = f.sigmoid() * c[layer] + i.sigmoid() * g.tanh()
            h[layer] = o.sigmoid() * c[layer].tanh()
            value = h[layer]
        outputs.append(p['output.weight'] @ value + p['output.bias'])
    return torch.stack(outputs)


def neural(dataset, seed):
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for saved-model replay')
    name = f'{dataset}-seed{seed}'
    result = json.loads((OUT / f'{name}.json').read_text())
    verify(result['inputs'])
    verify(result['trace'])
    training, validation, identity = load_inputs(dataset)
    if json.loads((ROOT / result['inputs']['path']).read_text()) != identity:
        raise ValueError('Input identity mismatch')
    trace = [json.loads(s) for s in (ROOT / result['trace']['path']).read_text().splitlines()]
    if len(trace) != 6000 or result['updates'] != 6000 or result['target_characters'] != 49_152_000:
        raise ValueError('Incomplete training exposure')
    for step, row in enumerate(trace, 1):
        if (row['step'] != step or row['target_characters'] != 8192 * step
                or row['learning_rate'] != learning_rate(step)
                or not all(math.isfinite(row[k]) and row[k] >= 0 for k in
                           ('training_nats_per_character', 'gradient_norm_before_clip', 'elapsed_seconds'))):
            raise ValueError('Invalid training trace')
    checkpoints = result['checkpoints']
    if tuple(r['step'] for r in checkpoints) != CHECKPOINTS:
        raise ValueError('Checkpoint inventory incomplete')
    for row in checkpoints:
        verify(row['checkpoint'])
        if json.loads((OUT / f'{name}-step{row["step"]}.json').read_text()) != row:
            raise ValueError('Checkpoint summary mismatch')
        score = row['selection']
        if (score['characters'] != 359276 or len(score['record_bits']) != len(validation)
                or not math.isclose(math.fsum(score['record_bits']), score['bits'], rel_tol=0, abs_tol=1e-8)):
            raise ValueError('Selection denominator mismatch')
    winner = min(checkpoints, key=lambda r: (r['selection']['bits_per_character'], r['step']))
    if winner != result['selected']:
        raise ValueError('Wrong source-only selection')
    payload = torch.load(ROOT / winner['checkpoint']['path'], weights_only=True, map_location='cpu')
    if payload['seed'] != seed or payload['dataset'] != dataset or payload['step'] != winner['step']:
        raise ValueError('Saved model identity mismatch')
    model = RecurrentSource(**payload['config']).eval()
    model.load_state_dict(payload['state_dict'])
    table = {c: i for i, c in enumerate(ALPHABET)}
    sample = [len(ALPHABET)] + [table[c] for c in validation[0][:127]]
    reference = manual_logits(model, sample)
    with torch.inference_mode():
        actual, _ = model(torch.tensor([sample]))
    manual_delta = float((reference - actual[0].double()).abs().max())
    if manual_delta > 2e-4:
        raise ValueError('Explicit double-precision LSTM gate replay failed')
    model.to('mps')
    # Different batch size and reloaded weights, all validation characters again.
    replay = score_records(model, validation, 'mps', batch=13)
    delta = abs(replay['bits_per_character'] - winner['selection']['bits_per_character'])
    per_record_delta = max(abs(a - b) / len(t) for a, b, t in
                           zip(replay['record_bits'], winner['selection']['record_bits'], validation))
    if delta > 1e-5 or per_record_delta > 1e-5:
        raise ValueError('Full saved-model selection replay failed')
    return {'run': name, 'selected_step': winner['step'], 'selected_bpc': replay['bits_per_character'],
            'bpc_replay_delta': delta, 'maximum_record_bpc_delta': per_record_delta,
            'manual_gate_maximum_logit_difference': manual_delta,
            'trace_rows': len(trace), 'checkpoint_hashes_verified': len(checkpoints),
            'result': artifact(OUT / f'{name}.json')}


def statistical(dataset):
    name = f'statistical-{dataset}'
    result = json.loads((OUT / f'{name}.json').read_text())
    training, validation, identity = load_inputs(dataset)
    verify(result['inputs'])
    if json.loads((ROOT / result['inputs']['path']).read_text()) != identity:
        raise ValueError('Statistical input identity mismatch')
    payload = load_archive(result['counts'])
    counts = payload['counts']
    if dict(Counter(''.join(training))) != counts['']:
        raise ValueError('Independent root counts failed')
    # Fixed stratified indices, selected without reference to selection losses.
    contexts = sorted(c for c in counts if c)
    sampled = [contexts[int(i)] for i in np.linspace(0, len(contexts) - 1, min(48, len(contexts)))]
    for context in sampled:
        expected = Counter()
        for text in training:
            start = 0
            while True:
                found = text.find(context, start)
                if found < 0:
                    break
                following = found + len(context)
                if following < len(text):
                    expected[text[following]] += 1
                start = found + 1
        if dict(expected) != counts[context]:
            raise ValueError('Independent overlapping-substring count failed')
    for context, row in counts.items():
        if context and (sum(row.values()) < 4 or context[1:] not in counts or context[:-1] not in counts):
            raise ValueError('Cutoff or closure failure')
    if tuple(r['tau'] for r in result['scores']) != TAUS:
        raise ValueError('Mass grid changed')
    winner = min(result['scores'], key=lambda r: (r['selection_bpc'], r['tau']))
    if winner != result['selected']:
        raise ValueError('Wrong statistical selection')
    tau = winner['tau']
    @lru_cache(maxsize=200000)
    def probabilities(history):
        if not history:
            row = counts['']
            return tuple((row.get(c, 0) + .5) / (sum(row.values()) + .5 * len(ALPHABET)) for c in ALPHABET)
        row = counts.get(history)
        lower = probabilities(history[1:])
        if row is None:
            return lower
        denominator = sum(row.values()) + tau
        return tuple((row.get(c, 0) + tau * p) / denominator for c, p in zip(ALPHABET, lower))
    indices = {c: i for i, c in enumerate(ALPHABET)}
    terms = []
    for _, text in chunks(validation):
        for i, char in enumerate(text):
            terms.append(-math.log2(probabilities(text[max(0, i - 12):i])[indices[char]]))
    bits = math.fsum(terms)
    delta = abs(bits - winner['selection_bits'])
    if delta > 1e-6:
        raise ValueError('Independent raw-history source score failed')
    return {'run': name, 'selected_tau': tau, 'selected_bpc': bits / len(terms),
            'replayed_characters': len(terms), 'selection_bits_delta': delta,
            'independent_context_count_checks': len(sampled) + 1,
            'contexts_verified_closed': len(counts), 'result': artifact(OUT / f'{name}.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('neural', 'statistical'))
    parser.add_argument('--dataset', choices=('small', 'large'), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    if (args.stage == 'neural') != (args.seed is not None):
        parser.error('Seed required only for neural audit')
    require_frozen(args.freeze, PATHS)
    name = f'{args.dataset}-seed{args.seed}' if args.stage == 'neural' else f'statistical-{args.dataset}'
    save_new(OUT / f'{name}-audit-started.json', {'freeze': args.freeze, 'time_unix': time.time()})
    limit_resources(900, 1200)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = neural(args.dataset, args.seed) if args.stage == 'neural' else statistical(args.dataset)
        result.update(status='PASS', resources=resource_report(wall, cpu), audit_freeze=args.freeze)
        save_new(OUT / f'{name}-audit.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-audit-failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
