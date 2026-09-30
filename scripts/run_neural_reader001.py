"""Fresh supplied-key reading after all source models and audits are frozen.

No stage may start until the original four fits and their audits are complete.
Prepare alone opens the reserved author archives; predict never loads answers.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import time
from dataclasses import asdict

import torch

from scripts.audit_suffix_reader001 import infer_reference, reading_score
from scripts.build_blind_channel_dev001 import make_channel
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_latin_source_compact001 import PATHS as COMPACT_PATHS
from scripts.run_latin_source_model001 import CORPUS, CORPUS_SHA, ROOT, SEEDS, artifact, limit_resources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.fresh_reader_panel import allocate_windows, encode_known, reader_gate, shuffled_plaintext
from voynich.recurrent_latin_source import ALPHABET, RecurrentSource, score_records
from voynich.recurrent_unit_beam import RecurrentProvider, decode_beam
from voynich.sparse_suffix_source import decode

EXP = 'NEURAL-READER-001'
OUT = ROOT / 'results' / EXP
BULK = ROOT / 'outputs' / EXP
AUTHORS = ('phi0588', 'phi1212')
ARMS = ('statistical-small', 'statistical-large', 'neural-31103-b32', 'neural-31103-b128',
        'neural-31109-b32', 'neural-31109-b128')
SOURCE_RESULTS = [f'results/LATIN-SOURCE-MODEL-001/{data}-seed{seed}{suffix}.json'
                  for data in ('small', 'large') for seed in SEEDS for suffix in ('', '-audit')]
SOURCE_RESULTS += [f'results/LATIN-SOURCE-COMPACT-001/{data}{suffix}.json'
                   for data in ('small', 'large') for suffix in ('', '-audit')]
PATHS = sorted(set(COMPACT_PATHS + SOURCE_RESULTS + [
    'scripts/run_neural_reader001.py', 'scripts/audit_suffix_reader001.py',
    'scripts/build_blind_channel_dev001.py', 'scripts/evaluate_naibbe001.py',
    'src/voynich/finite_state_channel.py', 'src/voynich/finite_state_channel_fit.py',
    'src/voynich/recurrent_unit_beam.py', 'src/voynich/compact_suffix_adapter.py',
    'src/voynich/compact_suffix_reference.py', 'src/voynich/fresh_reader_panel.py',
    'tests/test_recurrent_unit_beam.py', 'tests/test_compact_suffix_adapter.py',
    'tests/test_compact_suffix_reference.py', 'tests/test_fresh_reader_panel.py',
    'tests/test_neural_reader001.py', 'docs/experiments/NEURAL-READER-001.md',
]))


def verify(spec):
    if artifact(ROOT / spec['path']) != spec:
        raise ValueError('Bound artifact mismatch')


def sources():
    models = {}
    initializations = {}
    for data in ('small', 'large'):
        for seed in SEEDS:
            path = ROOT / f'results/LATIN-SOURCE-MODEL-001/{data}-seed{seed}.json'
            result = json.loads(path.read_text())
            audit = json.loads(path.with_name(path.stem + '-audit.json').read_text())
            if (result['status'] != 'completed_source_only' or audit['status'] != 'PASS'
                    or audit['result'] != artifact(path) or result['updates'] != 6000):
                raise ValueError('All source fits must be complete and audited')
            verify(result['selected']['checkpoint'])
            initializations[data, seed] = result['initialization_sha256']
            if data == 'large':
                models[f'neural-{seed}'] = {'checkpoint': result['selected']['checkpoint'],
                                           'result': artifact(path), 'audit': artifact(path.with_name(path.stem + '-audit.json'))}
        path = ROOT / f'results/LATIN-SOURCE-COMPACT-001/{data}.json'
        result = json.loads(path.read_text())
        audit_path = path.with_name(path.stem + '-audit.json')
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'PASS' or audit['result'] != artifact(path):
            raise ValueError('Compact source audit missing or mismatched')
        verify(result['counts'])
        models[f'statistical-{data}'] = {'counts': result['counts'], 'tau': result['selected']['tau'],
                                        'result': artifact(path), 'audit': artifact(audit_path)}
    if any(initializations['small', seed] != initializations['large', seed] for seed in SEEDS):
        raise ValueError('Paired initialization mismatch')
    return models


def prepare(freeze):
    selected = sources()  # Must precede reserved-author access.
    raw = (ROOT / CORPUS).read_bytes()
    if digest(raw) != CORPUS_SHA:
        raise ValueError('Corpus manifest changed')
    manifest = json.loads(raw)
    windows, author_artifacts = {}, {}
    for index, author in enumerate(AUTHORS):
        entry = manifest['authors'][author]
        if entry['role'] != 'reserved_reader_author':
            raise ValueError('Wrong fresh-author role')
        payload = load_archive(entry['artifact'])
        if payload['alphabet'] != ALPHABET or payload['author'] != author:
            raise ValueError('Reserved payload identity mismatch')
        windows[author] = allocate_windows(payload['records'], seed=571031 + 257 * index)
        author_artifacts[author] = entry['artifact']
    cases, answers, signatures = {}, {}, set()
    # Compare all prior CONFIRM001/SUFFIX001/SUFFIX002 key schedules without their plaintexts.
    for base, count in ((60129, 8), (88129, 16), (119129, 16)):
        for index in range(count):
            channel = make_channel(tuple(ALPHABET), 'B', base + 104729 * index)
            signatures.add(tuple(channel.rows['s0', c][0].glyphs for c in ALPHABET))
    for index in range(16):
        seed = 169129 + 104729 * index
        channel = make_channel(tuple(ALPHABET), 'B', seed)
        units = tuple(channel.rows['s0', c][0].glyphs for c in ALPHABET)
        if units in signatures:
            raise ValueError('Duplicate key; no redraw')
        signatures.add(units)
        plain = [windows[author][index]['text'] for author in AUTHORS]
        for null in (False, True):
            name = f'B-key{index + 1}' + ('-shuffle' if null else '')
            text = [shuffled_plaintext(value, seed + 4_000_000 + 257 * j) for j, value in enumerate(plain)] if null else plain
            cases[name] = {'positive': not null, 'units': units, 'key_seed': seed, 'authors': AUTHORS,
                           'records': [encode_known(value, ALPHABET, units) for value in text]}
            answers[name] = text
    panel = save_new(BULK / 'panel.json', {'alphabet': ALPHABET, 'cases': cases})
    answer = save_new(BULK / 'answers.json', {'cases': answers})
    validate_panel(cases)
    locations = {a: [{k: v for k, v in row.items() if k != 'text'} for row in rows] for a, rows in windows.items()}
    return {'experiment': EXP, 'source_freeze': freeze, 'sources': selected, 'panel': panel,
            'answers': answer, 'source_windows': locations, 'authors': author_artifacts,
            'known_keys_supplied': True, 'plaintext_permutation_nulls_have_known_answers': True,
            'positive_keys': 16, 'positive_records': 32, 'null_records': 32,
            'corpus_sha256': CORPUS_SHA}



def validate_panel(panel):
    expected = {f'B-key{i}' + suffix for i in range(1, 17) for suffix in ('', '-shuffle')}
    if set(panel) != expected:
        raise ValueError('Complete positive and null key panel required')
    for name, row in panel.items():
        if (row['positive'] != (not name.endswith('-shuffle')) or tuple(row['authors']) != AUTHORS
                or len(row['records']) != 2 or len(row['units']) != len(ALPHABET)
                or len(set(row['units'])) != len(ALPHABET)
                or any(not 1 <= len(u) <= 2 or set(u) - set('ABCDEF') for u in row['units'])
                or any(not 224 <= len(text) <= 448 or set(text) - set('ABCDEF') for text in row['records'])):
            raise ValueError('Panel assistance, support or dimensions changed')


def load_neural(spec):
    verify(spec['checkpoint'])
    payload = torch.load(ROOT / spec['checkpoint']['path'], weights_only=True, map_location='cpu')
    model = RecurrentSource(**payload['config'])
    model.load_state_dict(payload['state_dict'])
    if payload['config']['alphabet'] != ALPHABET:
        raise ValueError('Neural alphabet changed')
    return model.to('mps').eval()


def path_score(model, text):
    score = score_records(model, [text], 'mps', length=max(1, len(text)))
    return -score['bits'] * math.log(2) + len(text) * math.log1p(-1 / 225) + math.log(1 / 225)


def predict(freeze):
    require_frozen(freeze, [f'results/{EXP}/panel_manifest.json'])
    manifest = json.loads((OUT / 'panel_manifest.json').read_text())
    if sources() != manifest['sources']:
        raise ValueError('Sources changed after panel preparation')
    panel = checked_artifact(manifest['panel'])['cases']
    validate_panel(panel)
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for neural reading')
    output, max_delta = {}, {}
    for arm in ARMS:
        output[arm], max_delta[arm] = {}, 0.
        neural = arm.startswith('neural-')
        if neural:
            _, seed, width = arm.split('-')
            model = load_neural(manifest['sources'][f'neural-{seed}'])
            provider = RecurrentProvider(model, 'mps')
        else:
            spec = manifest['sources'][arm]
            verify(spec['counts'])
            compact = CompactSuffixSource.load(ROOT / spec['counts']['path'])
            source = CompactSuffixAdapter(compact, spec['tau'])
            raw = {'alphabet': ALPHABET, 'order': 12, 'tau': spec['tau'], 'counts': LiteralCountView(compact)}
        for name, case in panel.items():
            readings = []
            for observed in case['records']:
                if neural:
                    reading = asdict(decode_beam(provider, case['units'], observed, 1 / 225,
                                                 beam_width=int(width[1:]), max_expanded=200000))
                    if reading['plaintext'] is None:
                        raise ValueError('Generated complete support lost')
                    replay = path_score(model, reading['plaintext'])
                    delta = abs(replay - reading['joint_log_probability'])
                    if delta > 1e-4 * max(1, len(reading['plaintext'])):
                        raise ValueError('Full-forward versus incremental score mismatch')
                    if torch.mps.driver_allocated_memory() > 8 * 1024**3:
                        raise MemoryError('Reader 8GiB driver cap')
                else:
                    reading = decode(source, case['units'], observed, 1 / 225, max_nodes=500000).to_dict()
                    if reading['plaintext'] is None:
                        raise ValueError('Statistical reader lost generated support')
                    total, best, nodes = infer_reference(raw, case['units'], observed, 1 / 225)
                    replay = reading_score(raw, case['units'], observed, reading['plaintext'], 1 / 225)
                    delta = max(abs(total - reading['log_likelihood']), abs(best - reading['joint_log_probability']),
                                abs(replay - reading['joint_log_probability']))
                    if delta > 1e-7 or nodes != reading['reachable_nodes']:
                        raise ValueError('Separate reverse statistical inference mismatch')
                if encode_known(reading['plaintext'], ALPHABET, case['units']) != observed:
                    raise ValueError('Reading fails exact re-encoding')
                reading['path_replay_log_probability'] = replay
                reading['maximum_replay_delta'] = delta
                max_delta[arm] = max(max_delta[arm], delta)
                readings.append(reading)
            output[arm][name] = readings
            archive = save_new(BULK / arm / f'{name}.json.gz', {'readings': readings}, compressed=True)
            save_new(OUT / f'{arm}-{name}.json', {'arm': arm, 'case': name, 'predictions': archive})
            print(json.dumps({'arm': arm, 'case_complete': name}), flush=True)
        if neural:
            del provider, model
            torch.mps.empty_cache()
        else:
            del raw, source, compact
        gc.collect()
    predictions = save_new(BULK / 'predictions.json.gz', {'arms': output}, compressed=True)
    return {'experiment': EXP, 'prediction_freeze': freeze, 'predictions': predictions,
            'panel_manifest': artifact(OUT / 'panel_manifest.json'), 'records_per_arm': 64,
            'arms': ARMS, 'maximum_replay_deltas': max_delta, 'answers_not_loaded': True}


def edit_reference(a, b):
    # Full grid, independent of the older evaluation helper's rolling-row routine.
    table = [list(range(len(b) + 1))] + [[i] + [0] * len(b) for i in range(1, len(a) + 1)]
    for i, first in enumerate(a, 1):
        for j, second in enumerate(b, 1):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + (first != second))
    return table[-1][-1]


def evaluate(freeze):
    require_frozen(freeze, [f'results/{EXP}/{p}.json' for p in ('panel_manifest', 'prediction_manifest')])
    manifest = json.loads((OUT / 'panel_manifest.json').read_text())
    frozen = json.loads((OUT / 'prediction_manifest.json').read_text())
    verify(frozen['panel_manifest'])
    outputs = load_archive(frozen['predictions'])['arms']
    truth = checked_artifact(manifest['answers'])['cases']
    panel = checked_artifact(manifest['panel'])['cases']
    validate_panel(panel)
    if set(outputs) != set(ARMS) or len(truth) != 32:
        raise ValueError('Complete six-arm thirty-two-case panel required')
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for true-path diagnostic')
    if any(len(rows) != 2 or any(len(text) != 224 or set(text) - set(ALPHABET) for text in rows)
           for rows in truth.values()):
        raise ValueError('Answer dimensions changed')
    for name, answers in truth.items():
        if [encode_known(text, ALPHABET, panel[name]['units']) for text in answers] != panel[name]['records']:
            raise ValueError('True paths fail exact re-encoding')
    table, diagnostics = {}, {}
    for arm in ARMS:
        if set(outputs[arm]) != set(truth):
            raise ValueError('Missing case')
        table[arm] = {}
        for name in truth:
            rows = outputs[arm][name]
            if len(rows) != 2:
                raise ValueError('Missing author record')
            summary = metrics(rows, truth[name])
            independent = [edit_reference(r['plaintext'], answer) for r, answer in zip(rows, truth[name], strict=True)]
            if independent != summary['record_edits'] or sum(independent) != summary['edits']:
                raise ValueError('Independent edit-distance mismatch')
            table[arm][name] = summary
    for seed in SEEDS:
        arm = f'neural-{seed}-b128'
        model = load_neural(manifest['sources'][f'neural-{seed}'])
        diagnostics[arm] = {}
        for name, answers in truth.items():
            rows = []
            for i, answer in enumerate(answers):
                gold = path_score(model, answer)
                predicted = outputs[arm][name][i]
                narrow = outputs[f'neural-{seed}-b32'][name][i]
                tolerance = 1e-4 * max(len(answer), len(predicted['plaintext']), len(narrow['plaintext']))
                candidate = predicted['path_replay_log_probability']
                rows.append({'gold_path_log_probability': gold, 'primary_path_log_probability': candidate,
                             'narrow_path_log_probability': narrow['path_replay_log_probability'],
                             'gold_minus_primary': gold - candidate,
                             'better_known_path_proves_search_gap': gold > candidate + tolerance,
                             'wrong_returned_path_beats_gold': predicted['plaintext'] != answer and candidate > gold + tolerance,
                             'narrow_search_finds_better_score': narrow['path_replay_log_probability'] > candidate + tolerance,
                             'score_bound_flag_computed_not_interval_proof': predicted['score_bound_certifies_map']})
            diagnostics[arm][name] = rows
        del model
        torch.mps.empty_cache()
    positive = [f'B-key{i}' for i in range(1, 17)]
    errors = {arm: [table[arm][name]['edits'] for name in positive] for arm in ARMS}
    gates = {f'neural-{seed}-b128': reader_gate(errors[f'neural-{seed}-b128'], errors['statistical-large'])
             for seed in SEEDS}
    return {'experiment': EXP, 'evaluation_freeze': freeze, 'status': 'fresh_known_key_reading_not_key_discovery',
            'case_metrics': table, 'positive_total_edits': {arm: sum(row) for arm, row in errors.items()},
            'null_total_edits': {arm: sum(table[arm][name + '-shuffle']['edits'] for name in positive) for arm in ARMS},
            'positive_gold_characters': 7168, 'null_gold_characters': 7168, 'primary_gates': gates,
            'both_neural_seeds_pass': all(row['pass'] for row in gates.values()),
            'search_source_diagnostics': diagnostics,
            'null_metrics_are_diagnostics_not_language_identification': True,
            'prediction_manifest': artifact(OUT / 'prediction_manifest.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'predict', 'evaluate'))
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / f'{args.stage}-started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limits = {'prepare': (300, 300), 'predict': (3600, 3600), 'evaluate': (900, 1200)}
    limit_resources(*limits[args.stage])
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = {'prepare': prepare, 'predict': predict, 'evaluate': evaluate}[args.stage](args.freeze)
        result['resources'] = resource_report(wall, cpu)
        target = {'prepare': 'panel_manifest', 'predict': 'prediction_manifest', 'evaluate': 'evaluation'}[args.stage]
        save_new(OUT / f'{target}.json', result)
        print(json.dumps({'stage': args.stage, 'status': 'complete', 'resources': result['resources']}), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{args.stage}-failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
