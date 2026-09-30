"""Explicit fixed-shape execution revision of the immutable NEURAL-001 engine."""
import argparse
import json
import math
import time
from contextlib import contextmanager

import torch

from scripts import run_key_bank_neural001 as engine
from scripts.benchmark_sequence_memory001 import PATHS as BENCH_PATHS, check_memory, snapshot
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact
from voynich.fixed_sequence_scoring import fixed_record_logps

EXP = 'KEY-BANK-NEURAL-002'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
PATHS = sorted(set(BENCH_PATHS + [
    'scripts/run_key_bank_neural002.py', 'tests/test_key_bank_neural002.py',
    'docs/experiments/KEY-BANK-NEURAL-002.md',
    'results/SEQUENCE-MEMORY-001/campaign.json', 'results/SEQUENCE-MEMORY-001/fixed8.json',
    'results/SEQUENCE-MEMORY-001/audit.json',
    'results/KEY-BANK-NEURAL-001/prediction-failure.json',
    'results/KEY-BANK-NEURAL-001/evaluation.json']))


def admission(freeze):
    require_frozen(freeze, PATHS)
    base = ROOT/'results/SEQUENCE-MEMORY-001'
    report = checked_artifact(artifact(base/'fixed8.json'))
    audit = checked_artifact(artifact(base/'audit.json'))
    campaign = checked_artifact(artifact(base/'campaign.json'))
    process = [r for r in campaign['processes'] if r['arm'] == 'fixed8']
    if (report['status'] != 'PASS' or audit['status'] != 'PASS'
            or audit['campaign'] != artifact(base/'campaign.json')
            or audit['arms']['fixed8']['result'] != artifact(base/'fixed8.json')
            or len(process) != 1 or process[0]['returncode'] != 0
            or report['sampled_driver_peak'] > 2*1024**3
            or len(report['completed_models']) != 2
            or any(r['records'] != 4096 or r['maximum_float64_delta'] > 1e-3 for r in report['completed_models'])):
        raise ValueError('Matching successful fixed-shape memory qualification required')
    for row in report['completed_models']:
        if artifact(ROOT/row['scores']['path']) != row['scores']:
            raise ValueError('Benchmark score archive changed')
    if artifact(ROOT/report['trace']['path']) != report['trace']:
        raise ValueError('Benchmark trace changed')


class Scorer:
    """MPS-only shape revision; preserve the engine's independent CPU references."""
    def __init__(self, stream, cpu_reference):
        self.stream, self.cpu_reference = stream, cpu_reference
        self.wall, self.cpu = time.monotonic(), time.process_time()
        self.calls, self.records, self.characters, self.batches = 0, 0, 0, 0
        self.driver_peak, self.tensor_peak = 0, 0

    def __call__(self, model, texts, device):
        if str(device) == 'cpu':
            return self.cpu_reference(model, texts, device)
        if str(device) != 'mps':
            raise ValueError('Only explicitly registered CPU/MPS devices')
        self.calls += 1
        previous = 0
        def observe(info):
            nonlocal previous
            self.records += info['records']-previous
            previous = info['records']
            self.batches += 1
            sample = {**info, 'call': self.calls, 'total_records': self.records,
                **snapshot(), **resource_report(self.wall, self.cpu)}
            self.driver_peak = max(self.driver_peak, sample['driver_bytes'])
            self.tensor_peak = max(self.tensor_peak, sample['tensor_bytes'])
            self.stream.write(json.dumps(sample, allow_nan=False)+'\n')
            self.stream.flush()
            check_memory(sample)
        values = fixed_record_logps(model, texts, 'mps', batch=8, length=270, after_batch=observe)
        self.characters += sum(map(len, texts))
        return [v+len(t)*math.log1p(-1/225)+math.log(1/225)
                for t, v in zip(texts, values, strict=True)]


@contextmanager
def configured(scorer=None):
    """Scope only declared engine substitutions; restore even after failure.

    The original source, candidate lists, priors, checks, stopping limits and
    decisions are unchanged. Separate processes use a separate output namespace.
    """
    replacements = {'EXP': EXP, 'OUT': OUT, 'BULK': BULK, 'PATHS': PATHS}
    if scorer is not None:
        replacements['sequence_scores'] = scorer
    saved = {k: getattr(engine, k) for k in replacements}
    try:
        for k, v in replacements.items():
            setattr(engine, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(engine, k, v)


def run(mode, freeze):
    admission(freeze)
    if mode == 'evaluate':
        with configured():
            engine.evaluate(freeze)
        return
    if mode != 'predict':
        raise ValueError('Unknown stage')
    # The engine's exclusive start marker prevents repeating empirical attempts.
    # The trace is exclusive too; failure is never overwritten by a second run.
    BULK.mkdir(parents=True, exist_ok=True)
    with (BULK/'memory-trace.jsonl').open('x') as stream:
        scorer = Scorer(stream, engine.sequence_scores)
        try:
            with configured(scorer):
                engine.predict(freeze)
        finally:
            stream.flush()
            save_new(OUT/'memory-accounting.json', {
                'source_freeze': freeze, 'mps_calls_completed_or_attempted': scorer.calls,
                'mps_batch_records_processed': scorer.records, 'mps_batches': scorer.batches,
                'characters_in_completed_calls': scorer.characters,
                'sampled_driver_peak_bytes': scorer.driver_peak,
                'sampled_tensor_peak_bytes': scorer.tensor_peak,
                'trace': artifact(BULK/'memory-trace.jsonl'),
                'torch_version': str(torch.__version__),
                'not_continuous_memory_measurement': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('predict', 'evaluate'))
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    run(args.mode, args.freeze)
