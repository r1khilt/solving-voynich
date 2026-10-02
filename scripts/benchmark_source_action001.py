"""One frozen random-only shape benchmark; no Latin or cipher-panel access."""

import argparse
import copy
import gc
import json
import math
import signal
import time
from dataclasses import asdict

import torch

from scripts.run_source_state_systems001 import ROOT, artifact, limit_resources, require_frozen, resource_report, save_new
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal

EXP = 'SOURCE-ACTION-BENCH-001'
OUT = ROOT/'results'/EXP
CONFIGS = {
    'small': SourceActionConfig(width=256, heads=8, encoder_layers=4, decoder_layers=2, max_records=2),
    'large': SourceActionConfig(max_records=2)}
PARAMETERS = {'small': 5_775_918, 'large': 96_039_982}
PATHS = ['src/voynich/source_action_proposal.py', 'src/voynich/joint_key_proposal.py',
    'scripts/benchmark_source_action001.py', 'tests/test_source_action_proposal.py',
    'tests/test_source_action_benchmark.py', 'docs/experiments/SOURCE-ACTION-BENCH-001.md',
    'docs/research/source-action-inverse-2026-10-02.md', 'pyproject.toml', 'uv.lock']
WALL, CPU = 600, 500
HOST, DRIVER = 4*1024**3, 8*1024**3
SEED, BATCH, STEPS, WARMUP = 91811, 4, 5, 1


def fixture(config, *, batch=4, letters=224):
    if not 1 <= 2*letters <= config.max_glyphs:
        raise ValueError('Benchmark length must fit literal double-unit records')
    environments, traces = [], []
    units = config.glyphs+config.glyphs**2
    key = tuple(config.glyphs+(7*row)%(units-config.glyphs) for row in range(config.rows))
    for item in range(batch):
        texts = tuple(tuple((j*5+record*7+item)%config.rows for j in range(letters)) for record in range(2))
        dummy = ReadingEnvironment(((0,), (0,)), rows=config.rows, glyphs=config.glyphs)
        cipher = tuple(tuple(g for a in text for g in dummy.pool[key[a]]) for text in texts)
        environment = ReadingEnvironment(cipher, rows=config.rows, glyphs=config.glyphs)
        environments.append(environment)
        traces.append(environment.teaching_trace(texts, key))
    return environments, traces


def guard(wall, cpu):
    r = resource_report(wall, cpu)
    torch.mps.synchronize()
    driver, tensor = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
    if r['wall_seconds'] > WALL or r['cpu_seconds'] > CPU or r['peak_rss_bytes'] > HOST or driver > DRIVER:
        raise MemoryError('Prospective host/driver/time benchmark bound exceeded; no retry')
    return {**r, 'sampled_mps_driver_bytes': driver, 'sampled_mps_tensor_bytes': tensor}


def reference(model):
    environments, traces = fixture(model.config, batch=1, letters=6)
    cpu = copy.deepcopy(model).cpu().double().eval()
    with torch.no_grad():
        gpu_packed, cpu_packed = model.pack(environments, traces), cpu.pack(environments, traces)
        gpu = model(gpu_packed).cpu().double()
        exact = cpu(cpu_packed)
        mask = cpu_packed[5]
        delta = float((gpu[mask]-exact[mask]).abs().max())
        nll_delta = abs(float(model.loss(gpu_packed).item())-float(cpu.loss(cpu_packed).item()))
        if delta > .002 or nll_delta > .002:
            raise ArithmeticError('Short whole-action-path CPUdouble/MPS reference differs')
    del cpu
    gc.collect()
    return {'legal_logit_max_delta': delta, 'whole_path_nll_delta': nll_delta,
            'letters': 12, 'cipher_glyphs': 24, 'full_long_shape_reference_claimed': False}


def train_steps(model, packed, steps, *, sync=lambda: None, observe=lambda: None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, foreach=False)
    values = []
    for step in range(steps):
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(packed)
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite complete-path NLL')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise FloatingPointError('Missing/nonfinite action-model gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        optimizer.step()
        sync()
        values.append({'step': step+1, 'whole_path_nll': float(loss.item()), 'preclip_norm': norm,
                       'wall_seconds': time.monotonic()-began})
        observe()
    return values


def benchmark(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    resources, arms = [], []
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no CPU timing substitution')
        recommended = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(1., DRIVER/recommended))
        for name, config in CONFIGS.items():
            torch.manual_seed(SEED)
            model = SourceActionProposal(config).to('mps').eval()
            if sum(p.numel() for p in model.parameters()) != PARAMETERS[name]:
                raise ValueError('Prospective parameter count changed')
            resources.append(guard(wall, cpu))
            initial_reference = reference(model)
            resources.append(guard(wall, cpu))
            environments, traces = fixture(config, batch=BATCH, letters=224)
            began = time.monotonic()
            packed = model.pack(environments, traces)
            torch.mps.synchronize()
            packing_seconds = time.monotonic()-began
            model.train()
            def observe():
                resources.append(guard(wall, cpu))
            steps = train_steps(model, packed, STEPS, sync=torch.mps.synchronize, observe=observe)
            model.eval()
            final_reference = reference(model)
            resources.append(guard(wall, cpu))
            timed = [v['wall_seconds'] for v in steps[WARMUP:]]
            arm = {'arm': name, 'config': asdict(config), 'parameters': sum(p.numel() for p in model.parameters()),
                'batch': BATCH, 'records': 2, 'glyphs_per_record': 448, 'actions_per_episode': 448,
                'packing_seconds': packing_seconds, 'optimizer_steps': steps,
                'mean_timed_step_seconds': math.fsum(timed)/len(timed),
                'initial_reference': initial_reference, 'final_reference': final_reference}
            arms.append(arm)
            print(json.dumps({'arm': name, 'parameters': arm['parameters'],
                              'mean_step_seconds': arm['mean_timed_step_seconds']}), flush=True)
            del model, packed
            gc.collect()
            torch.mps.empty_cache()
        # Shape-only projection: four20k fits, two scales × two seeds. No
        # unmeasured inference/audit/preparation cost hidden in this estimate.
        projected = 2*20_000*sum(a['mean_timed_step_seconds']+a['packing_seconds'] for a in arms)
        save_new(OUT/'result.json', {'status': 'PASS_bounded_random_only_source_action_shapes',
            'freeze': freeze, 'frozen_paths': len(PATHS), 'inputs': [artifact(ROOT/p) for p in PATHS],
            'torch_version': torch.__version__, 'device': 'mps', 'cpu_threads': torch.get_num_threads(),
            'mps_recommended_working_set_bytes': recommended,
            'allocator_fraction': min(1., DRIVER/recommended),
            'seed': SEED, 'arms': arms, 'sampled_mps_driver_peak': max(r['sampled_mps_driver_bytes'] for r in resources),
            'sampled_host_peak': max(r['peak_rss_bytes'] for r in resources),
            'projected_four20k_shape_training_seconds': projected,
            'projection_includes_inference_audit_data_or_preparation': False,
            'latin_cipher_panel_or_holdout_files_opened': False, 'training_qualification': False,
            'substantive_language_training_launched': False, 'paid_spend_usd': 0,
            'resources': resource_report(wall, cpu)})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/'failure.json', {'error': repr(exc), 'freeze': freeze, 'completed_arms': arms,
            'sampled_resources': resources, 'resources': resource_report(wall, cpu), 'no_retry': True})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    benchmark(parser.parse_args().freeze)
