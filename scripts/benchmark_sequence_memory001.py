"""Artificial-only MPS batch/shape benchmark in fresh, sequential processes."""
import argparse
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_key_bank_neural001 import PATHS as OLD_PATHS
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.fixed_sequence_scoring import fixed_record_logps
from voynich.recurrent_latin_source import ALPHABET, RecurrentSource, score_records

EXP = 'SEQUENCE-MEMORY-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
ARMS = ('variable64', 'variable8', 'fixed8')
SEEDS = (71101, 71109)
PATHS = sorted(set(OLD_PATHS + [
    'src/voynich/fixed_sequence_scoring.py', 'scripts/benchmark_sequence_memory001.py',
    'tests/test_fixed_sequence_scoring.py', 'tests/test_sequence_memory001.py',
    'docs/experiments/SEQUENCE-MEMORY-001.md']))


def artificial_records():
    rng = np.random.default_rng(71001)
    return [''.join(rng.choice(list(ALPHABET), size=180+(i*37)%91)) for i in range(4096)]


def snapshot():
    torch.mps.synchronize()
    return {'driver_bytes': torch.mps.driver_allocated_memory(),
            'tensor_bytes': torch.mps.current_allocated_memory()}


def check_memory(sample):
    if sample['driver_bytes'] > 2*1024**3 or sample['peak_rss_bytes'] > 4*1024**3:
        raise MemoryError('Sampled 2GiB driver / 4GiB host guard')


def run_arm(arm, freeze):
    require_frozen(freeze, PATHS)
    if arm not in ARMS or not torch.backends.mps.is_available():
        raise ValueError('Registered arm and MPS access required')
    save_new(OUT/f'{arm}-started.json', {'freeze': freeze, 'start_unix': time.time(),
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'device': 'mps', 'cpu_numerical_threads': 2})
    limit_resources(180, 150)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    trace, completed = [], []
    texts = artificial_records()
    trace_path = BULK/f'{arm}-trace.jsonl'
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open('x') as stream:
        def measure(info):
            row = {**info, **snapshot(), **resource_report(wall, cpu)}
            trace.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            check_memory(row)
        try:
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = RecurrentSource().to('mps').eval()
                initial_digest = hashlib.sha256(b''.join(
                    t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
                measure({'seed': seed, 'stage': 'loaded', 'records': 0})
                began = time.monotonic()
                scores = []
                if arm == 'fixed8':
                    scores = fixed_record_logps(model, texts, 'mps', after_batch=lambda v: measure(
                        {'seed': seed, 'stage': 'scoring', **v}))
                else:
                    batch = 64 if arm == 'variable64' else 8
                    for start in range(0, len(texts), batch):
                        subset = texts[start:start+batch]
                        value = score_records(model, subset, 'mps', batch=batch, length=270)
                        if value['chunks'] != len(subset):
                            raise ValueError('Artificial histories were truncated')
                        scores.extend(-v*math.log(2) for v in value['record_bits'])
                        measure({'seed': seed, 'stage': 'scoring', 'records': len(scores),
                                 'batch_shape': [len(subset), max(map(len, subset))]})
                seconds = time.monotonic()-began
                indices = [i*(len(texts)-1)//15 for i in range(16)]
                reference = copy.deepcopy(model).to('cpu').double().eval()
                expected = score_records(reference, [texts[i] for i in indices], 'cpu', batch=8, length=270)
                delta = max(abs(scores[i]+v*math.log(2)) for i, v in zip(indices, expected['record_bits'], strict=True))
                if delta > 1e-3 or len(scores) != 4096 or any(not math.isfinite(v) for v in scores):
                    raise ValueError('Complete-score or float64 numerical gate failed')
                packed = save_new(BULK/f'{arm}-{seed}-scores.json.gz', {'record_logps': scores}, compressed=True)
                completed.append({'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                    'initial_weights_sha256': initial_digest, 'records': len(scores),
                    'characters': sum(map(len, texts)), 'scoring_wall_seconds': seconds,
                    'scores': packed, 'maximum_float64_delta': delta, 'reference_indices': indices})
                del model, reference
                torch.mps.empty_cache()
                measure({'seed': seed, 'stage': 'cleared', 'records': len(scores)})
            save_new(OUT/f'{arm}.json', {'status': 'PASS', 'arm': arm, 'freeze': freeze,
                'data_sha256': hashlib.sha256('\n'.join(texts).encode()).hexdigest(),
                'completed_models': completed, 'sampled_driver_peak': max(r['driver_bytes'] for r in trace),
                'sampled_tensor_peak': max(r['tensor_bytes'] for r in trace),
                'trace': artifact(trace_path), 'resources': resource_report(wall, cpu)})
        except Exception as error:
            signal.alarm(0)
            save_new(OUT/f'{arm}-failure.json', {'arm': arm, 'freeze': freeze,
                'error': str(error), 'type': type(error).__name__, 'last_sample': trace[-1] if trace else None,
                'completed_models': completed, 'trace': artifact(trace_path), 'resources': resource_report(wall, cpu)})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/'campaign-started.json', {'freeze': freeze, 'start_unix': time.time()})
    BULK.mkdir(parents=True, exist_ok=True)
    wall = time.monotonic()
    processes = []
    for arm in ARMS:
        began = time.monotonic()
        with (BULK/f'{arm}.log').open('x') as log:
            try:
                result = subprocess.run([sys.executable, '-u', __file__, 'arm', '--arm', arm, '--freeze', freeze],
                    cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, timeout=195, check=False)
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        row = {'arm': arm, 'returncode': code, 'outer_timeout': timed_out,
               'outer_wall_seconds': time.monotonic()-began, 'log': artifact(BULK/f'{arm}.log')}
        processes.append(row)
        print(json.dumps(row), flush=True)
    save_new(OUT/'campaign.json', {'freeze': freeze, 'processes': processes,
        'wall_seconds': time.monotonic()-wall, 'paid_spend_usd': 0,
        'only_artificial_inputs_and_untrained_models': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('campaign', 'arm'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    if args.mode == 'campaign':
        campaign(args.freeze)
    else:
        run_arm(args.arm, args.freeze)
