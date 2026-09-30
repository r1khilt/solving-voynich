"""Single-attempt, source-only paired Latin model fitting; no cipher panel access."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from voynich.recurrent_latin_source import (
    ALPHABET, RecurrentSource, WindowSampler, chunks, learning_rate, score_records,
)
from voynich.sparse_suffix_source import SuffixSource
from voynich.threshold_suffix_counts import collect_threshold_counts

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'LATIN-SOURCE-MODEL-001'
OUT = ROOT / 'results' / EXPERIMENT
BULK = ROOT / 'outputs' / EXPERIMENT
CORPUS = 'results/LATIN-SOURCE-001/corpus.json'
CORPUS_SHA = 'fcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5'
SEEDS = (31103, 31109)
CHECKPOINTS = (0, 100, 500, 1000, 2000, 4000, 6000)
TAUS = (16., 64., 256., 1024.)
PATHS = [
    'src/voynich/recurrent_latin_source.py', 'src/voynich/threshold_suffix_counts.py',
    'src/voynich/sparse_suffix_source.py', 'src/voynich/higher_order_unit_channel.py',
    'scripts/run_latin_source_model001.py', 'scripts/run_blind_channel_dev001.py',
    'scripts/run_blind_channel_dev004.py', 'scripts/audit_latin_source_model001.py',
    'tests/test_recurrent_latin_source.py', 'tests/test_threshold_suffix_counts.py',
    'tests/test_latin_source_model001.py', 'docs/experiments/LATIN-SOURCE-MODEL-001.md',
    CORPUS, 'results/LATIN-SOURCE-001/corpus_audit.json',
    'results/LATIN-SOURCE-001/overlap_audit.json', 'results/LATIN-SOURCE-001/hardware.json',
]
TRAINING_AUTHORS = ('phi0119', 'phi0134', 'phi0448', 'phi0550', 'phi0690', 'phi0836',
                    'phi0893', 'phi0959', 'phi1002', 'phi1017', 'phi1056', 'phi1348')


def artifact(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_author(manifest, author, role):
    if role not in ('training_pool', 'source_validation') or manifest['authors'][author]['role'] != role:
        raise ValueError('Reserved or wrong-role author access forbidden')
    if (role == 'training_pool' and author not in TRAINING_AUTHORS
            or role == 'source_validation' and author != 'phi1318'):
        raise ValueError('Author outside registered role inventory')
    payload = load_archive(manifest['authors'][author]['artifact'])
    if payload['alphabet'] != ALPHABET or payload['author'] != author:
        raise ValueError('Author payload mismatch')
    return payload['records']


def load_inputs(dataset):
    if dataset not in ('small', 'large'):
        raise ValueError('Unknown dataset')
    raw = (ROOT / CORPUS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA:
        raise ValueError('Frozen corpus manifest changed')
    manifest = json.loads(raw)
    roles = {a: r['role'] for a, r in manifest['authors'].items()}
    if (sorted(a for a, role in roles.items() if role == 'training_pool') != list(TRAINING_AUTHORS)
            or sorted(a for a, role in roles.items() if role == 'source_validation') != ['phi1318']):
        raise ValueError('Author roles changed')
    if dataset == 'small':
        payload = load_archive(manifest['small'])
        if payload['alphabet'] != ALPHABET:
            raise ValueError('Small alphabet mismatch')
        rows = payload['records']
        if any(r['author'] not in ('phi0448', 'phi0690') for r in rows):
            raise ValueError('Small author mismatch')
    else:
        rows = [r for a in TRAINING_AUTHORS for r in load_author(manifest, a, 'training_pool')]
    eligible = [r for r in rows if len(r['text']) >= 512]
    valid_rows = load_author(manifest, 'phi1318', 'source_validation')
    training = [r['text'] for r in eligible]
    validation = [r['text'] for r in valid_rows]
    identity = {'dataset': dataset, 'corpus_sha256': CORPUS_SHA,
                'characters_before_window_filter': sum(len(r['text']) for r in rows),
                'training_characters': sum(map(len, training)), 'training_segments': len(training),
                'training_legal_window_starts': sum(len(r) - 511 for r in training),
                'training_text_sha256': hashlib.sha256('\n'.join(training).encode()).hexdigest(),
                'validation_characters': sum(map(len, validation)), 'validation_segments': len(validation),
                'validation_text_sha256': hashlib.sha256('\n'.join(validation).encode()).hexdigest(),
                'eligible_records': [{'id': r['id'], 'author': r['author'],
                                     'sha256': r['sha256'], 'characters': len(r['text'])} for r in eligible]}
    expected = 97_850 if dataset == 'small' else 6_497_939
    if identity['training_characters'] != expected or identity['validation_characters'] != 359_276:
        raise ValueError('Frozen character totals changed')
    return training, validation, identity


def limit_resources(wall, cpu):
    def timeout(*_):
        raise TimeoutError('Registered wall limit; no restart or extension')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(wall)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def save_checkpoint(model, path, step, seed, dataset, freeze):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'config': model.config, 'state_dict': {k: v.detach().cpu().clone()
               for k, v in model.state_dict().items()}, 'step': step, 'seed': seed,
               'dataset': dataset, 'freeze': freeze, 'corpus_sha256': CORPUS_SHA}
    with path.open('xb') as handle:
        torch.save(payload, handle)
    return artifact(path)


def train(dataset, seed, freeze):
    if seed not in SEEDS:
        raise ValueError('Unregistered seed')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 100003)
    training, validation, identity = load_inputs(dataset)
    name = f'{dataset}-seed{seed}'
    save_new(OUT / f'{name}-inputs.json', identity)
    sampler = WindowSampler(training, 512)
    model = RecurrentSource().to('mps')
    initialization = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                            for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    wall, cpu = time.monotonic(), time.process_time()
    validations, last_loss = [], None
    trace_path = BULK / name / 'trace.jsonl'
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open('x') as trace:
        for step in range(6001):
            if step:
                model.train()
                x, y = sampler.sample(np_rng, 16)
                x, y = torch.from_numpy(x).to('mps'), torch.from_numpy(y).to('mps')
                rate = learning_rate(step)
                optimizer.param_groups[0]['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, len(ALPHABET)), y.reshape(-1))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.mps.synchronize()
                last_loss = float(loss.detach())
                row = {'step': step, 'training_nats_per_character': last_loss, 'learning_rate': rate,
                       'gradient_norm_before_clip': float(gradient), 'target_characters': step * 8192,
                       'elapsed_seconds': time.monotonic() - wall}
                trace.write(json.dumps(row, allow_nan=False) + '\n')
                if step % 50 == 0:
                    trace.flush()
                    print(json.dumps({'run': name, **row}), flush=True)
                if torch.mps.driver_allocated_memory() > 8 * 1024**3:
                    raise MemoryError('Registered 8GiB MPS driver cap exceeded')
            if step in CHECKPOINTS:
                score = score_records(model, validation, 'mps')
                if not math.isfinite(score['bits_per_character']):
                    raise FloatingPointError('Nonfinite selection loss')
                checkpoint = save_checkpoint(model, BULK / name / f'step{step}.pt', step, seed, dataset, freeze)
                row = {'step': step, 'checkpoint': checkpoint, 'selection': score,
                       'elapsed_seconds': time.monotonic() - wall}
                validations.append(row)
                save_new(OUT / f'{name}-step{step}.json', row)
                print(json.dumps({'run': name, 'checkpoint': step, 'pliny_bpc': score['bits_per_character']}), flush=True)
    winner = min(validations, key=lambda r: (r['selection']['bits_per_character'], r['step']))
    result = {'experiment': EXPERIMENT, 'status': 'completed_source_only', 'dataset': dataset,
              'seed': seed, 'freeze': freeze, 'inputs': artifact(OUT / f'{name}-inputs.json'),
              'initialization_sha256': initialization, 'parameters': sum(p.numel() for p in model.parameters()),
              'updates': 6000, 'target_characters': 6000 * 8192, 'trace': artifact(trace_path),
              'selected': winner, 'checkpoints': validations, 'last_training_nats_per_character': last_loss,
              'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
              'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory(),
              'resources': resource_report(wall, cpu)}
    save_new(OUT / f'{name}.json', result)
    print(json.dumps({'run': name, 'status': result['status'], 'selected_step': winner['step'],
                      'selection_bpc': winner['selection']['bits_per_character'], 'resources': result['resources']}), flush=True)


def statistics(dataset, freeze):
    torch.set_num_threads(1)
    wall, cpu = time.monotonic(), time.process_time()
    training, validation, identity = load_inputs(dataset)
    name = f'statistical-{dataset}'
    save_new(OUT / f'{name}-inputs.json', identity)
    counts = collect_threshold_counts(training, ALPHABET)
    if sum(counts[''].values()) != sum(map(len, training)):
        raise ValueError('Root counts do not match corpus')
    count_archive = save_new(BULK / name / 'counts.json.gz', {'alphabet': ALPHABET, 'order': 12,
                             'minimum': 4, 'counts': counts}, compressed=True)
    validation_chunks = [r for _, r in chunks(validation)]
    scores = []
    for tau in TAUS:
        model = SuffixSource(ALPHABET, 12, tau, counts)
        bits = model.bits(validation_chunks)
        row = {'tau': tau, 'selection_bits': bits, 'selection_bpc': bits / sum(map(len, validation))}
        scores.append(row)
        print(json.dumps({'run': name, **row, 'contexts': len(counts)}), flush=True)
        del model
        gc.collect()
    winner = min(scores, key=lambda r: (r['selection_bpc'], r['tau']))
    result = {'experiment': EXPERIMENT, 'status': 'completed_source_only', 'dataset': dataset,
              'freeze': freeze, 'inputs': artifact(OUT / f'{name}-inputs.json'), 'counts': count_archive,
              'contexts': len(counts), 'minimum_context_total': 4, 'order': 12, 'scores': scores,
              'selected': winner, 'resources': resource_report(wall, cpu)}
    save_new(OUT / f'{name}.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'statistics'))
    parser.add_argument('--dataset', choices=('small', 'large'), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    if args.stage == 'train' and args.seed is None or args.stage == 'statistics' and args.seed is not None:
        parser.error('Seed required only for train stage')
    require_frozen(args.freeze, PATHS)
    name = f'{args.dataset}-seed{args.seed}' if args.stage == 'train' else f'statistical-{args.dataset}'
    # Claim attempt before any empirical input loading or fitting. Never overwrite.
    save_new(OUT / f'{name}-started.json', {'freeze': args.freeze, 'stage': args.stage,
                                          'dataset': args.dataset, 'seed': args.seed, 'start_unix': time.time()})
    limit_resources(2400 if args.stage == 'train' else 1800, 4800 if args.stage == 'train' else 1800)
    try:
        if args.stage == 'train':
            train(args.dataset, args.seed, args.freeze)
        else:
            statistics(args.dataset, args.freeze)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'message': str(error),
                                              'freeze': args.freeze, 'time_unix': time.time()})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
