"""Replay exposed fixed keys with audited sources; never learn or repair a key."""
from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import time
from dataclasses import asdict

import torch

from scripts.audit_blind_channel_dev004 import literal_model_bits
from scripts.audit_suffix_reader001 import infer_reference, reading_score
from scripts.run_blind_channel_confirm001 import SOURCE_PATHS as ORIGINAL_PATHS
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import get_units, metrics, resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from scripts.run_neural_reader001 import PATHS as READER_PATHS
from scripts.run_neural_reader001 import edit_reference, load_neural, path_score, sources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.finite_state_channel import Channel
from voynich.fresh_reader_panel import encode_known
from voynich.recurrent_latin_source import ALPHABET
from voynich.recurrent_unit_beam import RecurrentProvider, decode_beam
from voynich.sparse_suffix_source import decode

EXP = 'KEY-SOURCE-DIAG-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
ORIGINAL = 'results/BLIND-CHANNEL-CONFIRM-001'
MANIFEST = 'data/manifests/blind_channel_confirm001.json'
NAMES = tuple(f'B-key{i}' + suffix for i in range(1, 9) for suffix in ('', '-shuffle'))
ARMS = ('statistical-small', 'statistical-large', 'neural-31103', 'neural-31109')
PATHS = sorted(set(READER_PATHS + ORIGINAL_PATHS + [
    MANIFEST, 'data/manifests/blind_channel_confirmation_corpora.json',
    f'{ORIGINAL}/evaluation.json', 'scripts/run_key_source_diag001.py',
    'scripts/audit_blind_channel_dev004.py', 'tests/test_key_source_diag001.py',
    'docs/experiments/KEY-SOURCE-DIAG-001.md',
] + [f'{ORIGINAL}/{name}_freeze.json' for name in NAMES]))


def support(units, observed):
    reachable = {0}
    for offset in range(len(observed)):
        if offset in reachable:
            reachable.update(offset + len(unit) for unit in units if observed.startswith(unit, offset))
    return len(observed) in reachable


def summarize_rows(rows, truth, model_bits, neural):
    result = metrics(rows, truth)
    if truth is not None:
        independent = [edit_reference(row['plaintext'] or '', gold) for row, gold in zip(rows, truth, strict=True)]
        if independent != result['record_edits']:
            raise ValueError('Independent per-record edit mismatch')
    supported = all(row['plaintext'] is not None for row in rows)
    result['model_bits'] = model_bits
    result['all_supported'] = supported
    if neural:
        lower = math.fsum(row['joint_log_probability'] for row in rows) if supported else None
        upper = math.fsum(max(row['joint_log_probability'], row['discarded_completion_upper_bound']
                             if row['discarded_completion_upper_bound'] is not None else row['joint_log_probability'])
                          for row in rows) if supported else None
        result.update(joint_map_lower_nats=None if lower is None else lower - model_bits * math.log(2),
                      joint_map_upper_nats=None if upper is None else upper - model_bits * math.log(2),
                      objective_kind='bounded_joint_map_not_marginal',
                      marginal_likelihood_available=False)
    else:
        value = math.fsum(row['log_likelihood'] for row in rows) if supported else None
        result.update(log_likelihood=value, total_bits=None if value is None else model_bits - value / math.log(2),
                      objective_kind='exact_path_marginal_plus_literal_code')
    return result


def compare_keys(learned, gold, neural):
    if not learned['all_supported'] or not gold['all_supported']:
        return {'status': 'support_differs', 'learned_supported': learned['all_supported'],
                'gold_supported': gold['all_supported']}
    if not neural:
        difference = learned['total_bits'] - gold['total_bits']
        return {'learned_minus_gold_total_bits': difference, 'true_key_is_known_better_candidate': difference > 1e-7}
    # Numerical guard scales with source lengths; these remain floating bounds.
    tolerance = 1e-4 * max(1, learned['decoded_characters'], gold['decoded_characters'])
    return {'gold_minus_learned_returned_joint_nats': gold['joint_map_lower_nats'] - learned['joint_map_lower_nats'],
            'floating_bounds_favor_gold': gold['joint_map_lower_nats'] > learned['joint_map_upper_nats'] + tolerance,
            'floating_bounds_favor_learned': learned['joint_map_lower_nats'] > gold['joint_map_upper_nats'] + tolerance,
            'not_marginal_or_interval_arithmetic_proof': True}


def load_cases():
    manifest = json.loads((ROOT / MANIFEST).read_text())
    require_frozen(manifest['pipeline_freeze'], ORIGINAL_PATHS)
    if set(manifest['cases']) != set(NAMES):
        raise ValueError('Original sixteen-case panel changed')
    cases = {}
    for name in NAMES:
        spec = manifest['cases'][name]
        chosen = json.loads((ROOT / ORIGINAL / f'{name}_freeze.json').read_text())
        if chosen['fit_sha256'] != spec['artifacts']['fit']['sha256']:
            raise ValueError('Original selected key has different fitting inputs')
        positive = not name.endswith('-shuffle')
        if spec['positive'] != positive:
            raise ValueError('Original case label changed')
        data = {split: checked_artifact(spec['artifacts'][split]) for split in ('fit', 'transfer')}
        if (len(data['fit']['records']) != 4 or len(data['transfer']['records']) != 2
                or data['fit']['context'] != data['transfer']['context']
                or ''.join(data['fit']['context']['source_alphabet']) != ALPHABET
                or data['fit']['context']['stop_probability'] != 1 / 225):
            raise ValueError('Original case dimensions/assistance changed')
        units = {'learned': tuple(chosen['units'])}
        truth = None
        if positive:
            answer = checked_artifact(spec['artifacts']['answer'])
            truth = answer['plaintext']
            units['gold'] = get_units(Channel.from_dict(answer['gold_channel']), tuple(ALPHABET))
            for split in data:
                if (len(truth[split]) != len(data[split]['records'])
                        or any(len(t) != 224 for t in truth[split])
                        or [encode_known(t, ALPHABET, units['gold']) for t in truth[split]] != data[split]['records']):
                    raise ValueError('Original gold path does not match ciphertext')
        cases[name] = {'data': data, 'units': units, 'truth': truth}
    return cases


def run(freeze):
    selected = sources()
    cases = load_cases()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; no silent fallback')
    reports, archives, maximum, count = {}, {}, {}, 0
    for arm in ARMS:
        neural = arm.startswith('neural')
        reports[arm], archives[arm], maximum[arm] = {}, {}, 0.
        if neural:
            model = load_neural(selected[arm])
            provider = RecurrentProvider(model, 'mps')
        else:
            spec = selected[arm]
            compact = CompactSuffixSource.load(ROOT / spec['counts']['path'])
            source = CompactSuffixAdapter(compact, spec['tau'])
            raw = {'alphabet': ALPHABET, 'order': 12, 'tau': spec['tau'], 'counts': LiteralCountView(compact)}
        for name, case in cases.items():
            report, full = {}, {}
            context = case['data']['fit']['context']
            for key, units in case['units'].items():
                report[key], full[key] = {}, {}
                code = literal_model_bits(units, context)
                for split, data in case['data'].items():
                    rows = []
                    for observed in data['records']:
                        possible = support(units, observed)
                        if neural:
                            row = asdict(decode_beam(provider, units, observed, 1 / 225,
                                                    beam_width=128, max_expanded=200000))
                            if row['plaintext'] is not None:
                                replay = path_score(model, row['plaintext'])
                                delta = abs(replay - row['joint_log_probability'])
                                if delta > 1e-4 * max(1, len(row['plaintext'])):
                                    raise ValueError('Neural path replay mismatch')
                                row['full_forward_log_probability'] = replay
                            else:
                                delta = 0.
                            if torch.mps.driver_allocated_memory() > 8 * 1024**3:
                                raise MemoryError('Registered GPU driver cap')
                        else:
                            row = decode(source, units, observed, 1 / 225, max_nodes=500000).to_dict()
                            total, best, nodes = infer_reference(raw, units, observed, 1 / 225)
                            if (math.isfinite(total) != possible or nodes != row['reachable_nodes']):
                                raise ValueError('Independent support/node mismatch')
                            delta = 0.
                            if possible:
                                replay = reading_score(raw, units, observed, row['plaintext'], 1 / 225)
                                delta = max(abs(total - row['log_likelihood']), abs(best - row['joint_log_probability']),
                                            abs(replay - row['joint_log_probability']))
                                if delta > 1e-7:
                                    raise ValueError('Statistical marginal/MAP/path replay mismatch')
                        if (row['plaintext'] is not None) != possible:
                            raise ValueError('Independent Boolean support mismatch')
                        if possible and encode_known(row['plaintext'], ALPHABET, units) != observed:
                            raise ValueError('Returned reading fails re-encoding')
                        row['replay_delta'] = delta
                        maximum[arm] = max(maximum[arm], delta)
                        rows.append(row)
                        count += 1
                    truth = None if case['truth'] is None else case['truth'][split]
                    report[key][split] = summarize_rows(rows, truth, code, neural)
                    full[key][split] = rows
            if 'gold' in report:
                report['fit_key_comparison'] = compare_keys(report['learned']['fit'], report['gold']['fit'], neural)
            archives[arm][name] = save_new(BULK / arm / f'{name}.json.gz', full, compressed=True)
            save_new(OUT / f'{arm}-{name}.json', {'source': arm, 'case': name, 'metrics': report,
                                                'readings': archives[arm][name]})
            reports[arm][name] = report
            print(json.dumps({'source': arm, 'case_complete': name}), flush=True)
        if neural:
            del provider, model
            torch.mps.empty_cache()
        else:
            del raw, source, compact
        gc.collect()
    if count != 576:
        raise ValueError('Incomplete predeclared panel')
    return {'experiment': EXP, 'freeze': freeze, 'status': 'exposed_fixed_key_diagnostic_not_qualification',
            'sources': selected, 'case_metrics': reports, 'archives': archives,
            'maximum_replay_deltas': maximum, 'reading_count': count, 'new_key_fits': 0,
            'original_evaluation': artifact(ROOT / ORIGINAL / 'evaluation.json'),
            'original_manifest': artifact(ROOT / MANIFEST)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / 'started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(900, 900)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = run(args.freeze)
        result['resources'] = resource_report(wall, cpu)
        save_new(OUT / 'evaluation.json', result)
        print(json.dumps({'status': 'complete', 'resources': result['resources']}), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
