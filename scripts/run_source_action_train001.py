"""Bounded paired reading-action training with explicit dictionary-memory control."""

import argparse
import copy
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from scripts import benchmark_source_action_cache001 as admission
from scripts import run_joint_key_train001 as old
from scripts.benchmark_key_proposal_systems001 import weights_digest
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.joint_key_training import EpisodeSampler, dictionary_code, metadata_bytes, schedule
from voynich.source_action_proposal import ReadingEnvironment, SourceActionProposal
from voynich.source_action_training import action_episode, control_metrics, control_records, greedy_reading, validation

EXP = 'SOURCE-ACTION-TRAIN-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
DEVICE = 'mps'
SEEDS = (92203, 92209)
ARMS = tuple(f'{role}-{seed}' for seed in SEEDS for role in ('binding', 'no-binding'))
STEPS, BATCH, VALIDATION_COUNT, VALIDATION_SEED = 20_000, 4, 64, 92221
CHECKPOINTS = (0, 1000, 4000, 10000, 20000)
CONTROL_COUNT, CONTROL_SEED = 16, 92251
CONFIG = admission.shapes.CONFIGS['large']
PARAMETERS = 96_039_982
FIT_WALL, FIT_CPU, CAMPAIGN_WALL = 28_800, 24_000, 118_800
HOST, DRIVER, BULK_LIMIT = 4*1024**3, 8*1024**3, 3*1024**3
BASE_PATHS = sorted(set([*admission.PATHS, *old.BASE_PATHS, *old.INPUT_PATHS,
    'results/JOINT-KEY-TRAIN-001/audit.json', 'results/SOURCE-ACTION-CACHE-001/result.json',
    'results/SOURCE-ACTION-CACHE-001/closed-check001.json',
    'src/voynich/source_action_training.py', 'tests/test_source_action_training.py',
    'src/voynich/unit_channel_decision.py', 'src/voynich/finite_state_channel.py',
    'scripts/run_source_action_train001.py', 'scripts/audit_source_action_train001.py',
    'tests/test_source_action_train001.py', 'docs/experiments/SOURCE-ACTION-TRAIN-001.md',
    'docs/research/source-action-training-design-2026-10-02.md']))
INPUT_PATHS = [f'results/{EXP}/{name}.json' for name in ('inputs', 'prepare', 'input-audit')]


def require_admission():
    value = json.loads((admission.OUT/'result.json').read_text())
    assert value['status'] == 'PASS_full_size_cached_action_gpu_admission'
    assert not (admission.OUT/'failure.json').exists() and len(value['cells']) == 16
    check = json.loads((admission.OUT/'closed-check001.json').read_text())
    assert check['result'] == artifact(admission.OUT/'result.json')
    assert check['status'] == 'PASS_cached_action_gpu_hash_resource_and_array_recheck'
    for spec in value['inputs']:
        assert artifact(ROOT/spec['path']) == spec
    assert json.loads((old.OUT/'audit.json').read_text())['status'] == 'PASS'


def prepared_payload():
    # Original role/body/edition/overlap audits remain binding. Use only their
    # audited training/Pliny pools; never open reserved authors or cipher panels.
    previous, texts, valid, old_episodes = old.load_prepared()
    expected_texts, expected_valid, identity = old.load_inputs('large')
    expected_valid = [r for r in expected_valid if len(r) >= old.MAX_LENGTH]
    assert texts == expected_texts and valid == expected_valid
    assert previous['identity']['training_text_sha256'] == identity['training_text_sha256']
    forbidden_raw = [dictionary_code(e[2]['raw_indices']) for e in old_episodes]
    forbidden_canonical = [dictionary_code(e[1]) for e in old_episodes]
    sampler = EpisodeSampler(valid, forbidden_raw=forbidden_raw, forbidden_canonical=forbidden_canonical)
    rng = np.random.default_rng(VALIDATION_SEED)
    episodes = [sampler.sample(rng) for _ in range(VALIDATION_COUNT)]
    assert len({dictionary_code(e[2]['raw_indices']) for e in episodes}) == VALIDATION_COUNT
    assert len({dictionary_code(e[1]) for e in episodes}) == VALIDATION_COUNT
    for episode in episodes:
        action_episode(sampler, episode)
    return previous, texts, valid, episodes, forbidden_raw, forbidden_canonical, sampler.rejected_keys


def prepare(freeze):
    old.require_frozen(freeze, BASE_PATHS)
    require_admission()
    save_new(OUT/'prepare-started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        previous, _, _, episodes, raw, canonical, rejected = prepared_payload()
        allocated = save_new(BULK/'validation-episodes.json.gz', {'metadata': [e[2] for e in episodes]}, compressed=True)
        manifest = {'freeze': freeze, 'parent_inputs': artifact(old.OUT/'inputs.json'),
            'parent_input_audit': artifact(old.OUT/'input-audit.json'), 'parent_full_audit': artifact(old.OUT/'audit.json'),
            'training': previous['training'], 'validation_source': previous['validation_source'],
            'validation_episodes': allocated, 'identity': previous['identity'],
            'old_forbidden_raw': raw, 'old_forbidden_canonical': canonical,
            'validation_seed': VALIDATION_SEED, 'validation_count': VALIDATION_COUNT,
            'new_model_records': 2, 'duplicate_records': False, 'preparation_rejected_keys': rejected}
        assert resource_report(wall, cpu)['peak_rss_bytes'] <= HOST
        save_new(OUT/'inputs.json', manifest)
        save_new(OUT/'prepare.json', {'status': 'PASS', 'freeze': freeze, 'inputs': artifact(OUT/'inputs.json'),
            'resources': resource_report(wall, cpu), 'paid_spend_usd': 0})
    except Exception as error:
        save_new(OUT/'prepare-failure.json', {'error': repr(error), 'no_retry': True})
        raise
    finally:
        signal.alarm(0)


def load_prepared():
    manifest = json.loads((OUT/'inputs.json').read_text())
    train, valid = load_archive(manifest['training']), load_archive(manifest['validation_source'])
    assert train['role'] == 'training_pool' and valid['role'] == 'source_validation'
    sampler = EpisodeSampler(valid['texts'])
    metadata = load_archive(manifest['validation_episodes'])['metadata']
    episodes = [sampler.make(e['windows'], e['raw_indices']) for e in metadata]
    assert [e[2] for e in episodes] == metadata and len(episodes) == VALIDATION_COUNT
    return manifest, train['texts'], valid['texts'], episodes


def audit_inputs(freeze):
    old.require_frozen(freeze, BASE_PATHS)
    save_new(OUT/'input-audit-started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest, texts, valid, episodes = load_prepared()
        previous, expected_train, expected_valid, expected, raw, canonical, rejected = prepared_payload()
        assert texts == expected_train and valid == expected_valid and episodes == expected
        assert manifest['identity'] == previous['identity'] and manifest['freeze'] == freeze
        assert manifest['old_forbidden_raw'] == raw and manifest['old_forbidden_canonical'] == canonical
        assert manifest['preparation_rejected_keys'] == rejected
        assert manifest['parent_inputs'] == artifact(old.OUT/'inputs.json')
        assert manifest['parent_input_audit'] == artifact(old.OUT/'input-audit.json')
        assert manifest['parent_full_audit'] == artifact(old.OUT/'audit.json')
        assert resource_report(wall, cpu)['peak_rss_bytes'] <= HOST
        save_new(OUT/'input-audit.json', {'status': 'PASS', 'freeze': freeze, 'inputs': artifact(OUT/'inputs.json'),
            'scope': 'exact source-role/body/edition/window/key/canonicalization/causal target construction',
            'resources': resource_report(wall, cpu), 'independent_agent_review': False, 'paid_spend_usd': 0})
    except Exception as error:
        save_new(OUT/'input-audit-failure.json', {'error': repr(error), 'no_retry': True})
        raise
    finally:
        signal.alarm(0)


def sync():
    if DEVICE == 'mps':
        torch.mps.synchronize()


def fit(arm, freeze):
    old.require_frozen(freeze, [*BASE_PATHS, *INPUT_PATHS])
    if arm not in ARMS or (DEVICE == 'mps' and (not torch.backends.mps.is_available()
                            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0')):
        raise ValueError('Registered arm and MPS with disabled CPU fallback required')
    save_new(OUT/(arm+'-started.json'), {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(FIT_WALL, FIT_CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    role, seed = arm.rsplit('-', 1)
    seed, binding = int(seed), role == 'binding'
    config = replace(CONFIG, binding_input=binding)
    BULK.mkdir(parents=True, exist_ok=True)
    rows, updates, episodes_seen, letters = [], 0, 0, 0
    data_digest = hashlib.sha256()
    ledger_path, trace_path = BULK/(arm+'-episodes.jsonl.gz'), BULK/(arm+'-trace.jsonl')
    with ledger_path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as ledger, trace_path.open('x') as trace:
        def observe(stage, **extra):
            sync()
            resource_row = resource_report(wall, cpu)
            driver = torch.mps.driver_allocated_memory() if DEVICE == 'mps' else 0
            owned = sum(p.stat().st_size for p in BULK.glob(arm+'-*') if p.is_file())
            if (resource_row['wall_seconds'] > FIT_WALL or resource_row['cpu_seconds'] > FIT_CPU
                    or resource_row['peak_rss_bytes'] > HOST or driver > DRIVER or owned > BULK_LIMIT):
                raise MemoryError('Prospective fit time/host/driver/bulk bound exceeded; no retry')
            trace.write(json.dumps({'stage': stage, 'driver_bytes': driver, 'owned_bulk_bytes': owned,
                                    **resource_row, **extra}, allow_nan=False)+'\n')
            trace.flush()
        try:
            manifest, texts, valid, episodes = load_prepared()
            audit = json.loads((OUT/'input-audit.json').read_text())
            assert audit['status'] == 'PASS' and audit['inputs'] == artifact(OUT/'inputs.json')
            raw_keys = manifest['old_forbidden_raw']+[dictionary_code(e[2]['raw_indices']) for e in episodes]
            canonical_keys = manifest['old_forbidden_canonical']+[dictionary_code(e[1]) for e in episodes]
            sampler = EpisodeSampler(texts, forbidden_raw=raw_keys, forbidden_canonical=canonical_keys)
            evaluator = EpisodeSampler(valid)
            rng = np.random.default_rng(seed+100003)
            torch.manual_seed(seed)
            if DEVICE == 'mps':
                torch.mps.set_per_process_memory_fraction(min(1., DRIVER/torch.mps.recommended_max_memory()))
            model = SourceActionProposal(config).to(DEVICE)
            assert sum(p.numel() for p in model.parameters()) == PARAMETERS
            initial = weights_digest(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, foreach=False)

            def snapshot(step):
                # The completed update has already consumed these gradients.
                # Avoid duplicating them in the unrelated CPU-double reference.
                optimizer.zero_grad(set_to_none=True)
                model.eval()
                short_env, short_trace = admission.fixture(config, 8, 'mixed')
                cpu_model = copy.deepcopy(model).cpu().double().eval()
                with torch.no_grad():
                    actual = model(model.pack([short_env], [short_trace])).detach().cpu().double()
                    reference = cpu_model(cpu_model.pack([short_env], [short_trace]))
                    mask = torch.isfinite(reference)
                    delta = float((actual[mask]-reference[mask]).abs().max())
                    if delta > .002 or not math.isfinite(delta):
                        raise ArithmeticError('Updated short action model differs from CPUdouble')
                del cpu_model
                gc.collect()
                score = validation(model, episodes, evaluator, batch=BATCH,
                    progress=lambda info: observe('validation', step=step, **info))
                controls = control_records(episodes, count=CONTROL_COUNT, seed=CONTROL_SEED)
                control_predictions = []
                for i, control in enumerate(controls):
                    control_predictions.append(greedy_reading(model,
                        ReadingEnvironment(control['records'], rows=config.rows, glyphs=config.glyphs)))
                    observe('null_control', step=step, control=i)
                score['controls'] = {'records': controls, 'predictions': control_predictions,
                                     'metrics': control_metrics(control_predictions, controls)}
                path = BULK/f'{arm}-step{step}.pt'
                with path.open('xb') as handle:
                    torch.save({'config': asdict(config), 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'step': step, 'seed': seed, 'freeze': freeze, 'inputs': artifact(OUT/'inputs.json')}, handle)
                score_spec = save_new(BULK/f'{arm}-step{step}-validation.json.gz',
                    {'score': score, 'short_cpu_double_legal_logit_delta': delta}, compressed=True)
                row = {'step': step, 'weights': artifact(path), 'weights_sha256': weights_digest(model),
                    'validation': score_spec, 'mean_whole_path_nll': score['mean_whole_path_nll'],
                    'recovery': score['recovery'], 'controls': score['controls']['metrics']}
                rows.append(row)
                save_new(OUT/f'{arm}-step{step}.json', row)
                observe('checkpoint', step=step)
                print(json.dumps({'arm': arm, 'step': step, 'whole_path_nll': row['mean_whole_path_nll'],
                    'exact_records': score['recovery']['exact_records'], 'complete_used_keys': score['recovery']['complete_used_keys']}), flush=True)
                model.train()

            snapshot(0)
            for step in range(1, STEPS+1):
                began = time.monotonic()
                batch = [sampler.sample(rng) for _ in range(BATCH)]
                metadata = metadata_bytes({'step': step, 'episodes': [e[2] for e in batch]})
                ledger.write(metadata)
                data_digest.update(metadata)
                episodes_seen += BATCH
                letters += sum(w['length'] for e in batch for w in e[2]['windows'])
                items = [action_episode(sampler, e) for e in batch]
                packed = model.pack([item[0] for item in items], [item[1] for item in items])
                optimizer.zero_grad(set_to_none=True)
                lr = schedule(step, total=STEPS)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                loss = model.loss(packed)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite whole-path training loss')
                loss.backward()
                if step in (1, STEPS):
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('Missing/nonfinite gradient')
                    if not binding and model.binding.weight.grad.abs().sum().item() != 0:
                        raise AssertionError('Binding-off neural input acquired a gradient')
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
                if not math.isfinite(norm):
                    raise FloatingPointError('Nonfinite preclip gradient norm')
                optimizer.step()
                sync()
                updates = step
                trace.write(json.dumps({'stage': 'optimizer', 'step': step, 'whole_path_nll': float(loss.item()),
                    'lr': lr, 'preclip_norm': norm, 'step_wall_seconds': time.monotonic()-began,
                    'episodes_seen': episodes_seen, 'source_letters': letters, 'data_sha256': data_digest.hexdigest()}, allow_nan=False)+'\n')
                if step % 100 == 0:
                    ledger.flush()
                    observe('resource', step=step)
                if step in CHECKPOINTS:
                    snapshot(step)
            selected = min(rows, key=lambda r: (r['mean_whole_path_nll'], r['step']))
            ledger.close()
            raw.flush()
            trace.flush()
            save_new(OUT/(arm+'.json'), {'status': 'PASS', 'arm': arm, 'seed': seed, 'freeze': freeze,
                'binding_input': binding, 'config': asdict(config), 'parameters': sum(p.numel() for p in model.parameters()),
                'inputs': artifact(OUT/'inputs.json'), 'updates': updates, 'episodes_seen': episodes_seen,
                'source_letters': letters, 'supervised_action_targets': letters, 'data_sha256': data_digest.hexdigest(),
                'initial_weights_sha256': initial, 'rejected_keys': sampler.rejected_keys,
                'ledger': artifact(ledger_path), 'trace': artifact(trace_path), 'checkpoints': rows, 'selected': selected,
                'resources': resource_report(wall, cpu), 'paid_spend_usd': 0, 'reserved_authors_or_cipher_panels_opened': False,
                'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                    'platform': platform.platform(), 'device': DEVICE, 'torch_threads': torch.get_num_threads(),
                    'omp_threads': os.environ.get('OMP_NUM_THREADS'), 'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
                    'mps_cpu_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}})
        except Exception as error:
            signal.alarm(0)
            ledger.close()
            raw.flush()
            trace.flush()
            save_new(OUT/(arm+'-failure.json'), {'error': repr(error), 'no_retry': True, 'completed_updates': updates,
                'generated_episodes': episodes_seen, 'source_letters': letters, 'checkpoints': rows,
                'ledger': artifact(ledger_path), 'trace': artifact(trace_path), 'resources': resource_report(wall, cpu)})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    old.require_frozen(freeze, [*BASE_PATHS, *INPUT_PATHS])
    if shutil.disk_usage(ROOT).free < 15*1024**3:
        raise OSError('At least 15GiB free disk required before the one campaign')
    save_new(OUT/'campaign-started.json', {'freeze': freeze, 'arms': ARMS, 'start_unix': time.time(), 'no_retry': True})
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    processes = []
    BULK.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        remaining = CAMPAIGN_WALL-(time.monotonic()-started)
        began = time.monotonic()
        with (BULK/(arm+'.log')).open('x') as log:
            if remaining <= 0:
                code, timeout = None, True
            else:
                try:
                    process = subprocess.run([sys.executable, '-u', '-m', 'scripts.run_source_action_train001',
                        'fit', '--arm', arm, '--freeze', freeze],
                        cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                        timeout=min(FIT_WALL+120, remaining), check=False)
                    code, timeout = process.returncode, False
                except subprocess.TimeoutExpired:
                    code, timeout = None, True
                except OSError as error:
                    log.write(repr(error)+'\n')
                    code, timeout = None, False
            row = {'arm': arm, 'returncode': code, 'outer_timeout': timeout,
                   'wall_seconds': time.monotonic()-began, 'log': artifact(BULK/(arm+'.log'))}
            processes.append(row)
            print(json.dumps(row), flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/'campaign.json', {'freeze': freeze, 'processes': processes, 'wall_seconds': time.monotonic()-started,
        'all_children_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime, 'paid_spend_usd': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'audit-inputs', 'fit', 'campaign'))
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    {'prepare': prepare, 'audit-inputs': audit_inputs, 'campaign': campaign}.get(args.mode,
        lambda freeze: fit(args.arm, freeze))(args.freeze)
