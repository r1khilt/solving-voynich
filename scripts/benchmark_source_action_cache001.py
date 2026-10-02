"""Frozen full-size cache equivalence/cost admission; artificial data only."""

import argparse
import copy
import gc
import json
import math
import os
import signal
import time
from dataclasses import asdict, replace

import torch

from scripts import benchmark_source_action001 as shapes
from scripts.run_source_state_systems001 import ROOT, artifact, limit_resources, require_frozen, resource_report, save_new
from voynich.source_action_cache import cached_trace_logits, propose_cached
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal

EXP = 'SOURCE-ACTION-CACHE-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
WALL, CPU, HOST, DRIVER = 600, 500, 4*1024**3, 8*1024**3
SEED = 92111
CASES = ((8, 'double'), (32, 'mixed'), (224, 'double'), (448, 'single'))
PATHS = sorted(set([*shapes.PATHS, 'src/voynich/source_action_cache.py',
    'tests/test_source_action_cache.py', 'scripts/benchmark_source_action_cache001.py',
    'tests/test_source_action_cache_benchmark.py', 'docs/experiments/SOURCE-ACTION-CACHE-001.md',
    'scripts/run_latin_source_model001.py', 'scripts/run_blind_channel_dev001.py',
    'scripts/run_blind_channel_dev004.py',
    'docs/research/source-action-cached-inference-2026-10-02.md',
    'results/SOURCE-ACTION-BENCH-001/result.json', 'results/SOURCE-ACTION-BENCH-001/closed-check001.json']))


def fixture(config, letters, family):
    if family not in ('double', 'mixed', 'single') or type(letters) is not int or letters < 1:
        raise ValueError('Declared finite artificial family required')
    pool = ReadingEnvironment(((0,),), rows=config.rows, glyphs=config.glyphs).pool
    key = tuple(config.glyphs+(7*r)%(len(pool)-config.glyphs) if family == 'double'
                else r%(config.glyphs if family == 'single' else min(len(pool), config.glyphs+5))
                for r in range(config.rows))
    lengths = (letters, letters if family != 'mixed' else letters//2+1)
    texts = tuple(tuple((j*5+record*7)%config.rows for j in range(n)) for record, n in enumerate(lengths))
    cipher = tuple(tuple(g for a in text for g in pool[key[a]]) for text in texts)
    env = ReadingEnvironment(cipher, rows=config.rows, glyphs=config.glyphs)
    return env, env.teaching_trace(texts, key)


def path_logq(logits, actions):
    value = logits.detach().cpu().double().log_softmax(-1)
    score = value[torch.arange(len(actions)), torch.tensor(actions)].sum().item()
    if not math.isfinite(score):
        raise FloatingPointError('Nonfinite complete-path density')
    return score


@torch.no_grad()
def compare(model, env, trace, *, sync=lambda: None):
    model.eval()
    began = time.monotonic()
    packed = model.pack([env], [trace])
    reference = model(packed)[0].detach().cpu().double()
    sync()
    full_seconds = time.monotonic()-began
    began = time.monotonic()
    cached = cached_trace_logits(model, env, trace).detach().cpu().double()
    sync()
    cache_seconds = time.monotonic()-began
    legal = packed[5][0].cpu()
    delta = float((reference[legal]-cached[legal]).abs().max())
    nll_delta = abs(path_logq(reference, trace[1])-path_logq(cached, trace[1]))
    if delta > .002 or nll_delta > .002:
        raise ArithmeticError('Cached/full complete-prefix logits or path density differ')
    if not torch.equal(torch.isneginf(reference), ~legal) or not torch.equal(torch.isneginf(cached), ~legal):
        raise ArithmeticError('Literal action masks differ')
    return {'legal_logit_max_delta': delta, 'whole_path_logq_delta': nll_delta,
        'reference_whole_path_logq': path_logq(reference, trace[1]),
        'cached_whole_path_logq': path_logq(cached, trace[1]),
        'full_teacher_forcing_wall_seconds': full_seconds, 'cached_sequential_wall_seconds': cache_seconds,
        'source_letters': len(trace[1]), 'record_glyph_lengths': list(map(len, env.records)),
        'sequential_uncached_speedup_claimed': False}, reference, cached


def guard(wall, cpu):
    torch.mps.synchronize()
    value = {**resource_report(wall, cpu), 'driver_bytes': torch.mps.driver_allocated_memory(),
             'tensor_bytes': torch.mps.current_allocated_memory()}
    if value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST or value['driver_bytes'] > DRIVER:
        raise MemoryError('Prospective cache benchmark bound exceeded; no retry')
    return value


def benchmark(freeze):
    require_frozen(freeze, PATHS)
    previous = json.loads((shapes.OUT/'result.json').read_text())
    assert previous['status'] == 'PASS_bounded_random_only_source_action_shapes'
    assert not (shapes.OUT/'failure.json').exists()
    save_new(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    cells, resources, bulk = [], [], 0
    try:
        if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
            raise RuntimeError('MPS with disabled CPU fallback required')
        torch.mps.set_per_process_memory_fraction(min(1., DRIVER/torch.mps.recommended_max_memory()))
        for name, config in shapes.CONFIGS.items():
            for bindings in (True, False):
                torch.manual_seed(SEED)
                model = SourceActionProposal(replace(config, binding_input=bindings)).to('mps').eval()
                assert sum(p.numel() for p in model.parameters()) == shapes.PARAMETERS[name]
                resources.append(guard(wall, cpu))
                for letters, family in CASES:
                    env, trace = fixture(config, letters, family)
                    metrics, full, cached = compare(model, env, trace, sync=torch.mps.synchronize)
                    reference_check = None
                    if letters == 8:
                        reference = copy.deepcopy(model).cpu().double().eval()
                        with torch.no_grad():
                            exact = reference(reference.pack([env], [trace]))[0].detach()
                        legal = torch.isfinite(exact)
                        reference_check = {'legal_logit_max_delta': float((cached[legal]-exact[legal]).abs().max()),
                            'whole_path_logq_delta': abs(path_logq(cached, trace[1])-path_logq(exact, trace[1]))}
                        assert max(reference_check.values()) <= .002
                        del reference
                        gc.collect()
                        for seed in (92131, 92137):
                            a, b = model.propose(env, seed=seed), propose_cached(model, env, seed=seed)
                            assert a['actions'] == b['actions'] and a['state'] == b['state'] and a['status'] == b['status']
                            assert abs(a['path_log_probability']-b['path_log_probability']) <= .002
                    def finite(tensor):
                        return [[float(v) if math.isfinite(float(v)) else None for v in row] for row in tensor]
                    archive = save_new(BULK/f'{name}-{int(bindings)}-{letters}-{family}.json.gz',
                        {'records': env.records, 'actions': trace[1], 'full_logits': finite(full), 'cached_logits': finite(cached)}, compressed=True)
                    bulk += archive['bytes']
                    if bulk > 128*1024**2:
                        raise MemoryError('128MiB private benchmark output cap')
                    cells.append({'scale': name, 'binding_input': bindings, 'config': asdict(model.config),
                        'letters_per_first_record': letters, 'family': family, 'metrics': metrics,
                        'short_cpu_double_check': reference_check, 'sampled_two_seeds_checked': letters == 8, 'archive': archive})
                    resources.append(guard(wall, cpu))
                    print(json.dumps({'scale': name, 'bindings': bindings, 'letters': letters, **metrics}), flush=True)
                del model
                gc.collect()
                torch.mps.empty_cache()
        # Finite one-row witness: the high-score single emission binds row0 to
        # glyph0, leaving glyph1 unsupported. No silent repair or refill.
        torch.manual_seed(SEED)
        fail_model = SourceActionProposal(SourceActionConfig(rows=1, glyphs=2, width=16, heads=2,
            encoder_layers=1, decoder_layers=1, max_glyphs=8)).to('mps')
        with torch.no_grad():
            fail_model.output.weight.zero_()
            fail_model.output.bias.copy_(torch.tensor([1000., -1000.], device='mps'))
        failed = propose_cached(fail_model, ReadingEnvironment(((0, 1),), rows=1, glyphs=2), seed=0)
        assert failed['status'] == 'dead_end' and failed['actions'] == (0,) and failed['state'].offsets == (1,)
        resources.append(guard(wall, cpu))
        assert len(cells) == 16
        require_frozen(freeze, PATHS)
        save_new(OUT/'result.json', {'status': 'PASS_full_size_cached_action_gpu_admission',
            'freeze': freeze, 'inputs': [artifact(ROOT/p) for p in PATHS], 'cells': cells,
            'forced_dead_end_preserved': True, 'bulk_bytes': bulk, 'torch': torch.__version__,
            'sampled_driver_peak': max(r['driver_bytes'] for r in resources),
            'sampled_host_peak': max(r['peak_rss_bytes'] for r in resources),
            'actual_training_or_latin_or_holdout_access': False, 'paid_spend_usd': 0,
            'resources': resource_report(wall, cpu), 'full_optimizer_or_independent_agent_audit_claimed': False})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/'failure.json', {'freeze': freeze, 'error': repr(error), 'cells': cells,
            'no_retry': True, 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    benchmark(parser.parse_args().freeze)
