"""Prospective known-key reader comparison; never reuses CONFIRM001 answers."""
from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import signal
import time
from pathlib import Path

from scripts.audit_suffix_reader001 import counts_reference, infer_reference, reading_score
from scripts.audit_blind_channel_dev004 import infer_record as old_reference
from scripts.build_blind_channel_dev001 import make_channel, segments
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from voynich.higher_order_unit_channel import MarkovSource, decode_units
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode

ROOT = Path(__file__).resolve().parents[1]
EXP = 'SUFFIX-READER-001'
DIR = ROOT / 'results' / EXP
CORPUS = 'data/manifests/blind_channel_development_corpora.json'
OLD_SELECTION = 'results/BLIND-CHANNEL-DEV-004/source_selection.json'
ORDERS, TAUS = (3, 5, 8, 12), (4., 16., 64., 256.)
PATHS = ['scripts/run_suffix_reader001.py', 'scripts/audit_suffix_reader001.py',
         'src/voynich/sparse_suffix_source.py', 'tests/test_sparse_suffix_source.py',
         'tests/test_suffix_reader001.py', 'docs/experiments/SUFFIX-READER-001.md',
         CORPUS, OLD_SELECTION, 'scripts/build_blind_channel_dev001.py',
         'scripts/run_blind_channel_dev001.py', 'scripts/run_blind_channel_dev004.py',
         'scripts/evaluate_naibbe001.py', 'src/voynich/higher_order_unit_channel.py',
         'src/voynich/finite_state_channel.py', 'src/voynich/finite_state_channel_fit.py',
         'data/manifests/blind_channel_dev001.json', 'scripts/audit_blind_channel_dev004.py']


def window_offsets():
    return [[40000 + 512 * i, 40256 + 512 * i] for i in range(16)]


def gate(edits, baseline):
    if len(edits) != 16 or len(baseline) != 16:
        raise ValueError('All sixteen keys required')
    return {'at_least_25pct_relative_edit_reduction': 4 * sum(edits) <= 3 * sum(baseline),
            'overall_cer_at_most_002': 50 * sum(edits) <= 16 * 448,
            'every_key_cer_at_most_005': all(20 * e <= 448 for e in edits)}


def start(stage, freeze, cpu, wall):
    require_frozen(freeze, PATHS)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    def timeout(*_):
        raise TimeoutError('Registered wall limit exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(wall)
    save_new(DIR / f'{stage}_started.json', {'freeze': freeze, 'start_unix': time.time()})
    return time.monotonic(), time.process_time()


def corpus_records(names):
    manifest = json.loads((ROOT / CORPUS).read_text())
    payloads = {}
    for name in names:
        spec = manifest['sources'][name]
        payloads[name] = checked_artifact({'path': spec['derived_path'], 'sha256': spec['derived_sha256']})
        if len(payloads[name]['text']) != 50000:
            raise ValueError('Pinned author length differs')
    return manifest, payloads


def prepare(freeze):
    wall, cpu = start('prepare', freeze, 1200, 1800)
    corpus, payloads = corpus_records(('caesar', 'virgil'))
    train, valid = segments(payloads['caesar']), segments(payloads['virgil'])
    alphabet = tuple(corpus['alphabet'])
    full_counts = collect_counts(train, alphabet, 12)
    if full_counts != counts_reference(train, 12):
        raise ValueError('Independent training counts differ')
    candidates = []
    for order in ORDERS:
        counts = {c: row for c, row in full_counts.items() if len(c) <= order}
        for tau in TAUS:
            source = SuffixSource(alphabet, order, tau, counts)
            value = {'order': order, 'tau': tau, 'contexts': len(source.contexts),
                     'validation_bits_per_character': source.bits(valid) / 50000}
            candidates.append(value)
            print(json.dumps(value), flush=True)
            del source
    chosen = min(candidates, key=lambda row: (row['validation_bits_per_character'], row['order'], row['tau']))
    del full_counts, counts
    gc.collect()
    counts = collect_counts(train + valid, alphabet, chosen['order'])
    if counts != counts_reference(train + valid, chosen['order']):
        raise ValueError('Independent refit counts differ')
    selected = SuffixSource(alphabet, chosen['order'], chosen['tau'], counts)
    selected_spec = save_new(ROOT / 'outputs' / EXP / 'source.json.gz', selected.to_dict(), compressed=True)
    baseline_spec = json.loads((ROOT / OLD_SELECTION).read_text())['models']
    baseline = MarkovSource.from_dict(load_archive(baseline_spec)['models']['order3'])
    baseline_counts = collect_counts(train + valid, alphabet, 3)
    sparse_baseline = SuffixSource(alphabet, 3, 256., baseline_counts)
    baseline_delta = max(abs(sparse_baseline.probabilities[sparse_baseline.state(h), j] - row[c])
                         for h, row in baseline.probabilities.items() for j, c in enumerate(alphabet))
    if baseline_delta > 1e-12:
        raise ValueError('Original baseline differs')
    selection = {'experiment': EXP, 'freeze': freeze, 'candidates': candidates, 'selected': chosen,
                 'source': selected_spec, 'source_contexts': len(selected.contexts), 'baseline': baseline_spec,
                 'baseline_regression_delta': baseline_delta, 'independent_all_counts_match': True,
                 'source_authors': ['caesar', 'virgil'], 'corpus_manifest_sha256': digest((ROOT / CORPUS).read_bytes())}
    save_new(DIR / 'source_selection.json', selection)
    # Only after source selection is saved do we access the target author.
    _, target = corpus_records(('cicero',))
    body = target['cicero']
    cases, answers, signatures, old_intervals = {}, {}, set(), []
    old_manifest = json.loads((ROOT / 'data/manifests/blind_channel_dev001.json').read_text())
    for name, spec in old_manifest['cases'].items():
        if spec['positive']:
            old = checked_artifact(spec['artifacts']['answer'])
            old_intervals.extend((offset, offset + 224) for offsets in old['source_offsets'].values() for offset in offsets)
    for index, offsets in enumerate(window_offsets()):
        seed = 88129 + 104729 * index
        key = make_channel(alphabet, 'B', seed)
        units = tuple(key.rows['s0', c][0].glyphs for c in alphabet)
        if units in signatures:
            raise ValueError('Duplicate key; no redraw')
        signatures.add(units)
        plain = []
        for offset in offsets:
            if any(offset < hi and offset + 224 > lo for lo, hi in old_intervals):
                raise ValueError('Target interval overlaps old cipher panel')
            if not any(lo <= offset and offset + 224 <= hi for lo, hi in
                       zip([0, *body['body_boundaries'][:-1]], body['body_boundaries'], strict=True)):
                raise ValueError('Target window crosses body boundary')
            value = body['text'][offset:offset + 224]
            if len(value) != 224:
                raise ValueError('Short target window')
            plain.append(value)
        name = f'B-key{index + 1}'
        cases[name] = {'units': units, 'records': [''.join(units[alphabet.index(c)] for c in p) for p in plain],
                       'key_seed': seed, 'offsets': offsets}
        answers[name] = plain
    panel = save_new(ROOT / 'data/processed' / EXP / 'panel.json', {'cases': cases, 'alphabet': alphabet})
    answer = save_new(ROOT / 'data/processed' / EXP / 'answers.json', {'cases': answers})
    save_new(DIR / 'panel_manifest.json', {'experiment': EXP, 'freeze': freeze, 'panel': panel,
                'answers': answer, 'target_author': 'cicero', 'target_derived_sha256': corpus['sources']['cicero']['derived_sha256'],
                'source_selection_sha256': digest((DIR / 'source_selection.json').read_bytes()),
                'known_keys_supplied': True, 'body_contained_nonoverlapping_old_panel': True,
                'resources': resource_report(wall, cpu)})


def predict(freeze):
    wall, cpu = start('predict', freeze, 1800, 2700)
    require_frozen(freeze, [f'results/{EXP}/source_selection.json', f'results/{EXP}/panel_manifest.json'])
    selection = json.loads((DIR / 'source_selection.json').read_text())
    manifest = json.loads((DIR / 'panel_manifest.json').read_text())
    if digest((DIR / 'source_selection.json').read_bytes()) != manifest['source_selection_sha256']:
        raise ValueError('Panel/source binding differs')
    panel = checked_artifact(manifest['panel'])
    raw = load_archive(selection['source'])
    selected = SuffixSource.from_dict(raw)
    baseline_raw = load_archive(selection['baseline'])['models']['order3']
    baseline = MarkovSource.from_dict(baseline_raw)
    output, delta = {}, 0.
    for name, case in panel['cases'].items():
        output[name] = {'baseline': [], 'selected': []}
        for observed in case['records']:
            old = decode_units(baseline, case['units'], observed, 1 / 225, max_nodes=500_000)
            old_check = old_reference(baseline_raw, tuple(case['units']), observed, 1 / 225, max_nodes=500_000)
            if old.plaintext is None or old_check['plaintext'] is None:
                raise ValueError('Baseline lost complete support')
            delta = max(delta, abs(old.log_likelihood - old_check['log_likelihood']),
                        abs(old.joint_log_probability - old_check['joint_log_probability']))
            new = decode(selected, case['units'], observed, 1 / 225, max_nodes=500_000)
            total, best, nodes = infer_reference(raw, case['units'], observed, 1 / 225)
            if new.plaintext is None or not math.isfinite(total):
                raise ValueError('Known complete singleton inventory lost support')
            delta = max(delta, abs(total - new.log_likelihood), abs(best - new.joint_log_probability),
                        abs(reading_score(raw, case['units'], observed, new.plaintext, 1 / 225) - best))
            if delta > 1e-7 or nodes != new.reachable_nodes:
                raise ValueError('Independent reverse recurrence differs')
            output[name]['baseline'].append(old.to_dict())
            output[name]['selected'].append(new.to_dict())
        partial = save_new(ROOT / 'outputs' / EXP / f'{name}.json.gz', output[name], compressed=True)
        save_new(DIR / f'{name}_prediction.json', {'case': name, 'predictions': partial})
        print(json.dumps({'completed': name}), flush=True)
    artifact = save_new(ROOT / 'outputs' / EXP / 'predictions.json.gz', {'cases': output}, compressed=True)
    save_new(DIR / 'prediction_manifest.json', {'experiment': EXP, 'freeze': freeze, 'predictions': artifact,
                'panel_manifest_sha256': digest((DIR / 'panel_manifest.json').read_bytes()),
                'source_selection_sha256': digest((DIR / 'source_selection.json').read_bytes()),
                'reverse_recurrence_records': 32, 'baseline_reference_records': 32, 'maximum_score_delta': delta,
                'resources': resource_report(wall, cpu)})


def evaluate(freeze):
    wall, cpu = start('evaluate', freeze, 120, 180)
    require_frozen(freeze, [f'results/{EXP}/{s}.json' for s in
                          ('prediction_manifest', 'panel_manifest', 'source_selection')])
    manifest = json.loads((DIR / 'panel_manifest.json').read_text())
    prediction_manifest = json.loads((DIR / 'prediction_manifest.json').read_text())
    if (digest((DIR / 'panel_manifest.json').read_bytes()) != prediction_manifest['panel_manifest_sha256']
            or digest((DIR / 'source_selection.json').read_bytes()) != prediction_manifest['source_selection_sha256']):
        raise ValueError('Evaluation provenance binding differs')
    predictions = load_archive(prediction_manifest['predictions'])['cases']
    truth = checked_artifact(manifest['answers'])['cases']
    if set(truth) != set(predictions) or len(truth) != 16:
        raise ValueError('Evaluation needs all keys')
    rows = {name: {arm: metrics(predictions[name][arm], truth[name]) for arm in ('baseline', 'selected')}
            for name in truth}
    checks = gate([r['selected']['edits'] for r in rows.values()], [r['baseline']['edits'] for r in rows.values()])
    result = {'experiment': EXP, 'freeze': freeze, 'status': 'known_key_new_allocation_development_not_blind_recovery',
              'cases': rows, 'gates': checks, 'reader_gate_pass': all(checks.values()),
              'total_edits': {arm: sum(r[arm]['edits'] for r in rows.values()) for arm in ('baseline','selected')},
              'gold_characters': 16 * 448, 'resources': resource_report(wall, cpu)}
    save_new(DIR / 'evaluation.json', result)
    print(json.dumps({k: result[k] for k in ('total_edits','gates','reader_gate_pass')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','predict','evaluate'))
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'predict': predict, 'evaluate': evaluate}[args.stage](args.freeze)
    except Exception as error:
        failure = DIR / f'{args.stage}_failure.json'
        if not failure.exists():
            save_new(failure, {'stage': args.stage, 'freeze': args.freeze,
                              'error_type': type(error).__name__, 'error': str(error)})
        raise
