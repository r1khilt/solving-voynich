"""Score-free, evaluator-owned construction; never imports a fitting function."""
import argparse
import hashlib
import json
import random
import signal
import time
from collections import Counter
from dataclasses import asdict

from scripts.build_blind_channel_dev001 import make_channel
from scripts.confirm002_common import (ALPHABET, BULK, CORPUS, DATA, EXP, FIT_PANEL, NAMES,
                                      OUT, PANEL, PREVIOUS_READER, ROOT, source_admission)
from scripts.run_blind_channel_dev001 import checked_artifact
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.finite_state_channel_fit import CodingContext
from voynich.fresh_cipher_panel import allocate_fresh, window_hashes
from voynich.fresh_reader_panel import encode_known

AUTHORS = ('phi0588', 'phi1212')
KEY_BASE, KEY_STRIDE = 269129, 104729
WINDOW_SEEDS = (671031, 671288)
PERMUTATION_SEED = 670003


def prior_signatures():
    signatures = set()
    schedules = [(47203, 1, 1), (47304, 1, 1), (60129, 8, KEY_STRIDE),
                 (88129, 16, KEY_STRIDE), (119129, 16, KEY_STRIDE), (169129, 16, KEY_STRIDE)]
    for base, count, stride in schedules:
        for index in range(count):
            channel = make_channel(tuple(ALPHABET), 'B', base+stride*index)
            signatures.add(tuple(channel.rows['s0', c][0].glyphs for c in ALPHABET))
    return signatures


def construct(windows):
    """Fixed 16-key schedule, opaque permutation, and glyph-count matched shuffles."""
    if len(windows['fit']) != 64 or len(windows['transfer']) != 32:
        raise ValueError('Fixed fresh window allocation required')
    inventory, signatures = [], prior_signatures()
    for index in range(16):
        seed = KEY_BASE+KEY_STRIDE*index
        channel = make_channel(tuple(ALPHABET), 'B', seed)
        units = tuple(channel.rows['s0', c][0].glyphs for c in ALPHABET)
        if units in signatures:
            raise ValueError('Key repeats a previous panel; no redraw')
        signatures.add(units)
        selected = {role: rows[index*count:(index+1)*count] for role, rows, count in
                    (('fit', windows['fit'], 4), ('transfer', windows['transfer'], 2))}
        texts = {role: [w['text'] for w in rows] for role, rows in selected.items()}
        if any(len(t) != 224 or set(t)-set(ALPHABET) for rows in texts.values() for t in rows):
            raise ValueError('Invalid normalized fresh text')
        encoded = {role: [encode_known(t, ALPHABET, units) for t in rows] for role, rows in texts.items()}
        context = asdict(CodingContext(tuple(ALPHABET), tuple('ABCDEF'), 32, 2, 2, 3, stop_probability=1/225))
        for positive in (True, False):
            rng = random.Random(seed+7_000_000)
            records = {}
            for role in ('fit', 'transfer'):
                records[role] = []
                for text in encoded[role]:
                    letters = list(text)
                    if not positive:
                        rng.shuffle(letters)
                    records[role].append(''.join(letters))
            answer = {'plaintext': texts if positive else None, 'units': units if positive else None,
                      'key_seed': seed, 'positive': positive, 'pair_index': index,
                      'null_seed': None if positive else seed+7_000_000,
                      'source_windows': {role: [{k: v for k, v in w.items() if k != 'text'} for w in rows]
                                         for role, rows in selected.items()}}
            inventory.append({'positive': positive, 'pair_index': index, 'context': context,
                              'records': records, 'answer': answer})
    random.Random(PERMUTATION_SEED).shuffle(inventory)
    return dict(zip(NAMES, inventory, strict=True))


def audit_constructed(cases, payloads):
    """Direct plaintext-location, encoding and histogram checks; no solver scores."""
    seen, positives, locations = set(), {}, set()
    for name, case in cases.items():
        answer = case['answer']
        if case['positive']:
            units = answer['units']
            if (len(units) != len(ALPHABET) or len(set(units)) != len(ALPHABET)
                    or sorted(u for u in units if len(u) == 1) != list('ABCDEF') or tuple(units) in seen):
                raise ValueError('Generating key family/uniqueness mismatch')
            seen.add(tuple(units))
            positives[case['pair_index']] = case
            table = dict(zip(ALPHABET, units, strict=True))
            for role, author in zip(('fit', 'transfer'), AUTHORS, strict=True):
                for j, window in enumerate(answer['source_windows'][role]):
                    row = payloads[author]['records'][window['record_index']]
                    text = row['text'][window['offset']:window['offset']+224]
                    if (row['id'] != window['record_id'] or row['sha256'] != window['record_sha256']
                            or text != answer['plaintext'][role][j]
                            or hashlib.sha256(text.encode()).hexdigest() != window['sha256']
                            or ''.join(table[c] for c in text) != case['records'][role][j]):
                        raise ValueError('Independent location/encoding audit failed')
                    slot = author, row['id'], window['offset']
                    if slot in locations:
                        raise ValueError('Reused fresh plaintext slot')
                    locations.add(slot)
    for case in cases.values():
        if not case['positive']:
            parent = positives[case['pair_index']]
            for role in ('fit', 'transfer'):
                if any(Counter(a) != Counter(b) for a, b in zip(parent['records'][role], case['records'][role], strict=True)):
                    raise ValueError('Null length/histogram mismatch')
    if len(cases) != 32 or len(positives) != 16 or len(locations) != 96:
        raise ValueError('Incomplete panel')
    return {'pass': True, 'positive_keys': 16, 'matched_nulls': 16, 'fresh_source_windows': 96,
            'independent_encoding_records': 96, 'paired_histogram_records': 96}


def prepare(freeze):
    source_admission(freeze)  # Must precede new reserved-text/key construction.
    save_new(OUT/'prepare-started.json', {'source_freeze': freeze, 'start_unix': time.time()})
    limit_resources(900, 600)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        corpus = json.loads((ROOT/CORPUS).read_text())
        old = json.loads((ROOT/PREVIOUS_READER).read_text())
        payloads, excluded, protected, inputs, protected_texts = {}, {}, set(), {}, []
        for author in AUTHORS:
            entry = corpus['authors'][author]
            if entry['role'] != 'reserved_reader_author' or entry['artifact'] != old['authors'][author]:
                raise ValueError('Author identity/role changed')
            payload = load_archive(entry['artifact'])
            if payload['author'] != author or payload['alphabet'] != ALPHABET:
                raise ValueError('Reserved text identity mismatch')
            payloads[author] = payload
            excluded[author] = {w['record_id'] for w in old['source_windows'][author]}
            for w in old['source_windows'][author]:
                row = payload['records'][w['record_index']]
                if (row['id'] != w['record_id'] or row['sha256'] != w['record_sha256']
                        or hashlib.sha256(row['text'][w['offset']:w['offset']+224].encode()).hexdigest() != w['sha256']):
                    raise ValueError('Previous allocation source binding changed')
            for row in payload['records']:
                if row['id'] in excluded[author]:
                    protected.update(window_hashes(row['text']))
                    protected_texts.append(row['text'])
            inputs[author] = entry['artifact']
        # Exclude overlap with whole prior editions, not just their chosen passages.
        for filename in ('data/manifests/blind_channel_development_corpora.json',
                         'data/manifests/blind_channel_confirmation_corpora.json'):
            previous = json.loads((ROOT/filename).read_text())
            for author, entry in previous['sources'].items():
                spec = {'path': entry['derived_path'], 'sha256': entry['derived_sha256']}
                value = checked_artifact(spec)
                protected.update(window_hashes(value['text']))
                protected_texts.append(value['text'])
                inputs[author] = artifact(ROOT/spec['path'])
        windows, allocation = {}, {}
        for role, author, count, seed in zip(('fit', 'transfer'), AUTHORS, (64, 32), WINDOW_SEEDS, strict=True):
            rows, stats = allocate_fresh(payloads[author]['records'], excluded[author], protected, count=count, seed=seed)
            windows[role], allocation[role] = rows, stats
            for row in rows:
                protected.update(window_hashes(row['text']))
        cases = construct(windows)
        audit = audit_constructed(cases, payloads)
        # Separate pairwise scan, independent literal representation, before any fitting.
        strings = [r['text'] for role in ('fit', 'transfer') for r in windows[role]]
        literal_seen = set()
        for text in strings:
            pieces = {text[i:i+64] for i in range(len(text)-63)}
            if pieces & literal_seen:
                raise ValueError('Fresh records share a 64-character window')
            literal_seen.update(pieces)
        # Different representation and scan direction from the allocator.
        for text in protected_texts:
            if any(text[i:i+64] in literal_seen for i in range(len(text)-63)):
                raise ValueError('Independent literal protected/fresh overlap check failed')
        audit['independent_literal_overlap_scan_pass'] = True
        fit_cases, full_cases = {}, {}
        for i, name in enumerate(NAMES):
            case = cases[name]
            artifacts = {role: save_new(DATA/name/f'{role}.json', {'case_id': name, 'records': rows, 'context': case['context']})
                         for role, rows in case['records'].items()}
            artifacts['answer'] = save_new(DATA/name/'answer.json', case['answer'])
            fit_cases[name] = {'fit': artifacts['fit'], 'search_seed': 73101+257*i}
            full_cases[name] = {**artifacts, 'positive': case['positive'], 'pair_index': case['pair_index']}
        fit = save_new(ROOT/FIT_PANEL, {'experiment': EXP, 'status': 'prepared_fit_only', 'source_freeze': freeze,
                                      'cases': fit_cases})
        ledger = save_new(BULK/'window-ledger.json.gz', windows, compressed=True)
        result = {'experiment': EXP, 'source_freeze': freeze, 'fit_panel': fit, 'cases': full_cases,
                  'construction_audit': audit, 'allocation': allocation, 'source_inputs': inputs, 'ledger': ledger,
                  'previous_reader': artifact(ROOT/PREVIOUS_READER), 'corpus': artifact(ROOT/CORPUS),
                  'whole_previous_record_exclusions': {a: sorted(ids) for a, ids in excluded.items()},
                  'new_passages_and_keys_not_new_authors': True, 'procedural_not_cryptographic_isolation': True,
                  'resources': resource_report(wall, cpu)}
        if result['resources']['peak_rss_bytes'] > 3*1024**3:
            raise MemoryError('Preparation sampled3GiB limit')
        save_new(ROOT/PANEL, result)
        print(json.dumps({'status': 'prepared', 'cases': 32, 'allocation': allocation}), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/'prepare-failure.json', {'type': type(error).__name__, 'error': str(error), 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    prepare(parser.parse_args().freeze)
