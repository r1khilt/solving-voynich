import json
import copy
import math

import pytest

from scripts.benchmark_global_search_systems001 import CONFIG, PATHS, workloads
from scripts.audit_global_search_systems001 import trace
from voynich.tempered_unit_search import TemperedConfig, search_tempered


def test_artificial_inputs_reproduce_without_opening_any_empirical_cipher_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Artificial builder must not read files')
    monkeypatch.setattr('builtins.open', forbidden)
    first, second = workloads(), workloads()
    assert first == second
    assert len(first['pool']) == 42 and len(set(first['pool'])) == 42
    assert all(len(key) == len(set(key)) == 23 for key in first['variants'])
    assert {tuple(sorted(key)) for key in first['variants']} == {tuple(sorted(first['variants'][0]))}
    assert all(sum(len(u) == 1 for u in key) == 6 for key in first['variants'])
    mapping = dict(zip(first['context']['source_alphabet'], first['variants'][0], strict=True))
    assert all(len(r) == 224 for r in first['plain'])
    for plain, full, short in zip(first['plain'], first['full_records'], first['short_records'], strict=True):
        assert full == ''.join(mapping[c] for c in plain)
        assert short == ''.join(mapping[c] for c in plain[:32])
    json.dumps(first, allow_nan=False)
    assert CONFIG.max_scored_keys == 2048 and CONFIG.max_seconds == 120.
    assert all('BLIND-CHANNEL-CONFIRM' not in path and 'blind_channel_confirm' not in path for path in PATHS)
    assert not any('panel' in path or '_freeze.json' in path for path in PATHS)


def tiny_trace():
    context = {'source_alphabet': list('ab'), 'glyph_alphabet': list('xy'), 'source_count': 1,
               'max_states': 2, 'max_alternatives': 3, 'max_emission_length': 2}
    work = []
    def score(key):
        logs = [-3.-key.count('y'), -2.-key.count('y')]
        code = 1+2*3+sum(map(len, key))
        value = sum(logs)-code*math.log(2)
        work.append({'units': key, 'record_log_likelihoods': logs, 'model_bits': code,
                     'log_weight_unnormalized': value})
        return value
    result = search_tempered(score, [('x', 'y')], ('x', 'y'),
                            config=TemperedConfig(iterations=8, seed=13))
    return json.loads(json.dumps(result)), json.loads(json.dumps(work)), context


def test_alternate_trace_accounting_catches_corrupted_scores_mappings_and_acceptances():
    result, work, context = tiny_trace()
    counts = trace(result, work, context)
    assert counts['proposal_events'] == 64 and counts['exchange_events'] == 7
    for kind in ('score', 'mapping', 'acceptance', 'work'):
        broken, rows = copy.deepcopy(result), copy.deepcopy(work)
        if kind == 'score':
            broken['bank'][0]['score'] += 1
        elif kind == 'mapping':
            broken['events'][0]['new_units'] = ['xy']
        elif kind == 'acceptance':
            broken['events'][0]['accepted'] = not broken['events'][0]['accepted']
        else:
            rows.pop()
        with pytest.raises(AssertionError):
            trace(broken, rows, context)
