"""One bounded fresh-process fixed-geometry memory stress test per cache policy."""

import argparse
import copy
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from scripts import run_source_action_train001 as parent
from voynich.joint_key_proposal import unit_pool
from voynich.source_action_proposal import ReadingEnvironment, SourceActionProposal
from voynich.source_action_static_pack import static_pack

EXP = 'SOURCE-ACTION-STATIC-001'
OUT, BULK = parent.ROOT/'results'/EXP, parent.ROOT/'outputs'/EXP
ARMS, STEPS, BATCH = ('fixed-retain', 'fixed-clear'), 256, 4
WALL, CPU, HOST, DRIVER = 600, 500, 4*1024**3, 8*1024**3
BULK_LIMIT = 16*1024**2
SEED, DATA_SEED = 92303, 92321
PATHS = sorted(set([*parent.BASE_PATHS, *parent.INPUT_PATHS,
    'results/SOURCE-ACTION-TRAIN-001/campaign.json', 'results/SOURCE-ACTION-TRAIN-001/audit.json',
    'results/SOURCE-ACTION-TRAIN-001/closed-check001.json',
    'src/voynich/source_action_static_pack.py', 'tests/test_source_action_static_pack.py',
    'scripts/benchmark_source_action_static001.py', 'tests/test_source_action_static_benchmark.py',
    'docs/experiments/SOURCE-ACTION-STATIC-001.md',
    'docs/research/source-action-memory-geometry-2026-10-02.md']))


def fixture(rng, step, config, *, batch=BATCH, low=64, high=224):
    if not 1 <= low <= high <= config.max_glyphs//2:
        raise ValueError('Bounded fixture source lengths required')
    environments, traces = [], []
    for _ in range(batch):
        key = tuple(map(int, rng.integers(42, size=23)))
        lengths = [high, high] if step % 16 == 0 else rng.integers(low, high+1, size=2)
        texts = [tuple(map(int, rng.integers(23, size=int(n)))) for n in lengths]
        pool = unit_pool(6)
        records = [tuple(g for a in text for g in pool[key[a]]) for text in texts]
        env = ReadingEnvironment(records)
        environments.append(env)
        traces.append(env.teaching_trace(texts, key))
    return environments, traces


@torch.no_grad()
def compare(model):
    env, trace = parent.admission.fixture(model.config, 8, 'mixed')
    ref = copy.deepcopy(model).cpu().double().eval()
    was_training = model.training
    model.eval()
    try:
        a = model(static_pack(model, [env], [trace])).detach().cpu().double()[0, :len(trace[1])]
        b = ref(ref.pack([env], [trace]))[0]
        legal = torch.isfinite(b)
        assert torch.equal(torch.isfinite(a), legal)
        delta = float((a[legal]-b[legal]).abs().max())
        choices = torch.tensor(trace[1])
        indices = torch.arange(len(choices))
        loss_delta = abs(float(a.log_softmax(-1)[indices, choices].sum()-b.log_softmax(-1)[indices, choices].sum()))
        assert max(delta, loss_delta) <= .002
        return {'legal_logit_max_delta': delta, 'whole_path_nll_delta': loss_delta,
                'cpu_reference_dynamic_mps_static': True}
    finally:
        del ref
        gc.collect()
        model.train(was_training)


def run(arm, freeze):
    parent.old.require_frozen(freeze, PATHS)
    if arm not in ARMS or not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise ValueError('Registered MPS arm with fallback disabled required')
    parent.save_new(OUT/(arm+'-started.json'), {'freeze': freeze, 'arm': arm, 'no_retry': True, 'start_unix': time.time()})
    parent.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.mps.set_per_process_memory_fraction(min(1., DRIVER/torch.mps.recommended_max_memory()))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(DATA_SEED)
    BULK.mkdir(parents=True, exist_ok=True)
    trace_path = BULK/(arm+'-trace.jsonl')
    samples, updates, times = [], 0, []
    with trace_path.open('x') as stream:
        def observe(stage, step):
            torch.mps.synchronize()
            resource = parent.resource_report(wall, cpu)
            value = {'stage': stage, 'step': step, 'driver_bytes': torch.mps.driver_allocated_memory(),
                     'current_tensor_bytes': torch.mps.current_allocated_memory(),
                     'owned_bulk_bytes': sum(p.stat().st_size for p in BULK.glob(arm+'*') if p.is_file()), **resource}
            stream.write(json.dumps(value, allow_nan=False)+'\n')
            stream.flush()
            samples.append(value)
            if (value['driver_bytes'] > DRIVER or value['peak_rss_bytes'] > HOST
                    or value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['owned_bulk_bytes'] > BULK_LIMIT):
                raise MemoryError('Registered geometry stress resource bound exceeded')
        try:
            model = SourceActionProposal(parent.CONFIG).to('mps')
            assert sum(p.numel() for p in model.parameters()) == parent.PARAMETERS
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, foreach=False)
            initial = parent.weights_digest(model)
            first = compare(model)
            torch.mps.empty_cache()  # One initial reference-cleanup common to both arms.
            observe('initial', 0)
            for step in range(1, STEPS+1):
                began = time.monotonic()
                environments, traces = fixture(rng, step, model.config)
                packed = static_pack(model, environments, traces)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(packed)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite stress path loss')
                observe('forward', step)
                loss.backward()
                observe('backward', step)
                if step in (1, STEPS):
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                assert math.isfinite(norm)
                optimizer.step()
                observe('update', step)
                value = float(loss.item())
                optimizer.zero_grad(set_to_none=True)
                del loss, packed
                if arm == 'fixed-clear':
                    torch.mps.empty_cache()
                observe('cleanup', step)
                elapsed = time.monotonic()-began
                times.append(elapsed)
                updates = step
                stream.write(json.dumps({'stage': 'objective', 'step': step, 'whole_path_nll': value,
                                         'preclip_norm': norm, 'step_wall_seconds': elapsed}, allow_nan=False)+'\n')
                if step % 32 == 0:
                    print(json.dumps({'arm': arm, 'step': step, 'wall': time.monotonic()-wall,
                                      'driver_bytes': samples[-1]['driver_bytes']}), flush=True)
            final = compare(model)
            observe('final', STEPS)
            growth = max(v['driver_bytes'] for v in samples if v['step'] >= 129)-max(
                v['driver_bytes'] for v in samples if 16 <= v['step'] <= 128)
            assert growth <= 256*1024**2
            stream.flush()
            parent.save_new(OUT/(arm+'.json'), {'status': 'PASS_fixed_geometry_memory_stress', 'arm': arm,
                'freeze': freeze, 'inputs': [parent.artifact(parent.ROOT/p) for p in PATHS],
                'updates': updates, 'parameters': parent.PARAMETERS, 'initial_weights_sha256': initial,
                'final_weights_sha256': parent.weights_digest(model), 'initial_numeric': first, 'final_numeric': final,
                'second_half_driver_growth_bytes': growth, 'max_driver_bytes': max(v['driver_bytes'] for v in samples),
                'mean_step_seconds_after16': math.fsum(times[16:])/len(times[16:]), 'trace': parent.artifact(trace_path),
                'resources': parent.resource_report(wall, cpu), 'language_or_recovery_claimed': False, 'paid_spend_usd': 0})
        except Exception as error:
            signal.alarm(0)
            stream.flush()
            parent.save_new(OUT/(arm+'-failure.json'), {'error': repr(error), 'completed_updates': updates,
                'freeze': freeze, 'trace': parent.artifact(trace_path), 'resources': parent.resource_report(wall, cpu), 'no_retry': True})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    parent.old.require_frozen(freeze, PATHS)
    parent.save_new(OUT/'campaign-started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    BULK.mkdir(parents=True, exist_ok=True)
    processes = []
    for arm in ARMS:
        path = BULK/(arm+'.log')
        with path.open('x') as log:
            try:
                process = subprocess.run([sys.executable, '-u', '-m', 'scripts.benchmark_source_action_static001',
                    'run', '--arm', arm, '--freeze', freeze], cwd=parent.ROOT, env=dict(os.environ),
                    stdout=log, stderr=subprocess.STDOUT, timeout=WALL+120, check=False)
                code, timeout = process.returncode, False
            except subprocess.TimeoutExpired:
                code, timeout = None, True
        row = {'arm': arm, 'returncode': code, 'outer_timeout': timeout, 'log': parent.artifact(path)}
        processes.append(row)
        print(json.dumps(row), flush=True)
    passing = [p['arm'] for p in processes if p['returncode'] == 0 and not p['outer_timeout']]
    parent.save_new(OUT/'campaign.json', {'freeze': freeze, 'processes': processes, 'passing': passing,
        'prospective_preference': 'fixed-retain' if 'fixed-retain' in passing else ('fixed-clear' if passing else None),
        'no_automatic_language_training': True, 'paid_spend_usd': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('campaign', 'run'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    (campaign(args.freeze) if args.mode == 'campaign' else run(args.arm, args.freeze))
