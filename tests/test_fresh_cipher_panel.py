"""Artificial allocation/encoding fixtures; never generates registered fresh keys."""
import copy
import hashlib
import random

import pytest

from scripts import build_blind_channel_confirm002 as builder
from voynich.fresh_cipher_panel import allocate_fresh, window_hashes


def record(identity, text):
    return {'id': identity, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}


def test_fixed_allocation_excludes_whole_records_protected_and_previous_slots():
    rng = random.Random(981)
    records = [record(str(i), ''.join(rng.choices('abcdefgh', k=512))) for i in range(5)]
    protected = window_hashes(records[2]['text'][:64], 8)
    args = dict(count=12, seed=711, width=32, stride=64, overlap=8)
    selected, stats = allocate_fresh(records, {'1'}, protected, **args)
    assert (selected, stats) == allocate_fresh(records, {'1'}, protected, **args)
    used = set(protected)
    for row in selected:
        assert row['record_id'] != '1' and row['offset'] % 64 == 0
        assert not window_hashes(row['text'], 8) & used
        used.update(window_hashes(row['text'], 8))
        original = records[row['record_index']]
        assert original['text'][row['offset']:row['offset']+32] == row['text']
    assert stats['candidate_slots'] == 32


def test_insufficient_windows_never_falls_back_to_excluded_data():
    records = [record('old', 'abcdefgh'*32), record('new', 'ijklmnop'*32)]
    with pytest.raises(ValueError, match='Insufficient'):
        allocate_fresh(records, {'old'}, window_hashes(records[1]['text']), count=1, seed=1)
    records[0]['text'] += 'x'
    with pytest.raises(ValueError, match='hash'):
        allocate_fresh(records, {'old'}, set(), count=1, seed=1)


@pytest.mark.parametrize('change', [{'overlap': 0}, {'overlap': None}, {'count': True},
                                  {'stride': 1}, {'width': 12}, {'count': 0}])
def test_invalid_allocation_fails(change):
    kwargs = dict(count=1, seed=1)
    kwargs.update(change)
    with pytest.raises(ValueError):
        allocate_fresh([record('a', 'a'*300)], set(), set(), **kwargs)


def test_artificial_construction_pairing_identity_and_tamper_detection(monkeypatch):
    # Deliberately NOT the registered qualification's key/window seeds.
    monkeypatch.setattr(builder, 'KEY_BASE', 900_100_001)
    monkeypatch.setattr(builder, 'PERMUTATION_SEED', 900_100_003)
    rng = random.Random(900_100_009)
    payloads, windows = {}, {}
    for role, author, count in zip(('fit', 'transfer'), builder.AUTHORS, (64, 32), strict=True):
        rows = [record(str(i), ''.join(rng.choices(builder.ALPHABET, k=224))) for i in range(count)]
        payloads[author] = {'records': rows}
        windows[role] = [{'text': r['text'], 'record_id': r['id'], 'record_index': i, 'offset': 0,
                          'record_sha256': r['sha256'], 'sha256': r['sha256']} for i, r in enumerate(rows)]
    cases = builder.construct(windows)
    assert builder.audit_constructed(cases, payloads)['fresh_source_windows'] == 96
    assert set(cases) == set(builder.NAMES)
    assert any(cases[n]['positive'] != (i % 2 == 0) for i, n in enumerate(builder.NAMES))
    null = next(c for c in cases.values() if not c['positive'])
    assert null['answer']['plaintext'] is None
    null['records']['fit'][0] += 'A'
    with pytest.raises(ValueError, match='histogram'):
        builder.audit_constructed(cases, payloads)
    bad = copy.deepcopy(windows)
    bad['fit'].pop()
    with pytest.raises(ValueError, match='allocation'):
        builder.construct(bad)


def test_key_collision_fails_without_redrawing(monkeypatch):
    monkeypatch.setattr(builder, 'KEY_BASE', 900_100_001)
    original = builder.make_channel
    channel = original(tuple(builder.ALPHABET), 'B', 900_100_001)
    units = tuple(channel.rows['s0', c][0].glyphs for c in builder.ALPHABET)
    monkeypatch.setattr(builder, 'prior_signatures', lambda: {units})
    with pytest.raises(ValueError, match='no redraw'):
        builder.construct({'fit': [None]*64, 'transfer': [None]*32})
