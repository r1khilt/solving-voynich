import copy
import json
import math
import gzip
import hashlib
from dataclasses import asdict

import pytest

from scripts.audit_global_key_search001 import validate_trace
from scripts import audit_global_key_search001 as auditor
from scripts.run_global_key_search001 import BoundedScorer, config_for, reader_union, warm_starts
from voynich.local_key_bank import one_move_bank
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode
from voynich.tempered_unit_search import TemperedConfig, search_tempered


def row(key, score):
    return {'units': list(key), 'model_bits': sum(map(len, key)), 'log_weight_unnormalized': score}


def test_warm_starts_keep_cold_baseline_and_rank_only_supported_distinct_endpoints():
    values = {('x', 'x'): 0., ('x', 'y'): 3., ('y', 'x'): 2., ('y', 'y'): 1.}
    starts, scores = warm_starts([('x', 'y'), ('y', 'x'), ('x', 'y'), ('y', 'y')], ('x', 'x'),
                                lambda key: row(key, values[key]))
    assert starts == [('x', 'x'), ('x', 'y'), ('y', 'x'), ('y', 'y')]+[('x', 'x')]*4
    assert len(scores) == 4
    with pytest.raises(ValueError, match='lost support'):
        warm_starts([('x', 'y')], ('x', 'x'), lambda key: row(key, None if key[1] == 'y' else 0.))


def test_whole_old_bank_preserved_bounded_new_retention_and_no_visit_multiplicity():
    old = {'bank': [row(('x', 'x'), 0.), row(('y', 'y'), None)]}
    visited = [row(('x', 'y'), 3.), row(('y', 'x'), 2.)]
    neighborhood = [row(('x', 'y'), 3.), row(('xy', 'x'), 4.)]
    result = reader_union(old, visited, neighborhood, keep=1)
    assert [r['units'] for r in result['bank']] == [['x', 'x'], ['y', 'y'], ['x', 'y'], ['xy', 'x']]
    assert result['bank_size'] == 4 and result['finite_keys'] == 3
    assert result['best_units'] == ['xy', 'x'] and result['retained_new_top_count'] == 1
    assert not result['visit_counts_used_as_weights'] and result['whole_old_bank_preserved']
    assert sum(math.exp(r['log_weight']) for r in result['bank'] if r['log_weight'] is not None) == pytest.approx(1.)
    for r in result['bank']:
        if r['units'] == ['x', 'y']:
            assert math.exp(r['log_weight']) == pytest.approx(math.exp(3)/(1+math.exp(3)+math.exp(4)))


@pytest.mark.parametrize('bad', [row(('x', 'x'), 1.), row(('x', 'x'), None)])
def test_duplicate_dictionary_score_disagreement_fails(bad):
    with pytest.raises(ValueError, match='inconsistent'):
        reader_union({'bank': [row(('x', 'x'), 0.)]}, [bad], [])


def test_tied_retention_uses_first_visitation_and_repeated_key_has_single_weight():
    old = {'bank': [row(('x', 'x'), 0.)]}
    result = reader_union(old, [row(('y', 'x'), 0.), row(('x', 'y'), 0.)], [], keep=1)
    assert [r['units'] for r in result['bank']] == [['x', 'x'], ['y', 'x']]
    assert result['best_index'] == 0


def test_native_wrapper_forwards_registered_large_graph_caps():
    class Native:
        def score(self, *args, **kwargs): return args, kwargs
    args, kwargs = BoundedScorer(Native()).score(('x', 'y'), 'xy', .25)
    assert args == (('x', 'y'), 'xy', .25)
    assert kwargs == {'max_nodes': 10_000_000, 'max_edges': 40_000_000}
    assert len({config_for(f'case-{i:02d}').seed for i in range(1, 33)}) == 32


def test_new_trace_accounting_handles_multiple_unique_starts_and_rejects_corruptions():
    cfg = TemperedConfig(seed=43, temperatures=(1., 3., 10.), iterations=25, block_sizes=(2,))
    def score(key): return None if key == ('y', 'y') else float(key.count('x'))
    result = json.loads(json.dumps(search_tempered(score, [('x', 'x'), ('x', 'y'), ('y', 'x')], ('x', 'y'), config=cfg)))
    counts = validate_trace(result, cfg)
    assert counts['proposals'] == 75
    for field in ('best_score', 'cache_hits', 'completed_iterations'):
        broken = copy.deepcopy(result)
        broken[field] += 1
        with pytest.raises(AssertionError):
            validate_trace(broken, cfg)


def test_full_auditor_artificial_file_lifecycle_and_independent_reverse_replay(tmp_path, monkeypatch):
    """Real archived schemas/hash bindings, with no empirical input or native call."""
    out = tmp_path/'results'
    out.mkdir()
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/audit_global_key_search001.py').write_bytes(
        (auditor.ROOT/'scripts/audit_global_key_search001.py').read_bytes())
    def identity(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    def save(path, value, compressed=False):
        payload = json.dumps(value, allow_nan=False).encode()
        path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
        return identity(path)
    cfg = TemperedConfig(seed=123, iterations=8, block_sizes=(2,))
    source = SuffixSource('ab', 1, 4., collect_counts(['abbaabaaaa', 'bbbaba'], 'ab', 1))
    context = {'source_alphabet': list('ab'), 'glyph_alphabet': list('xy'), 'stop_probability': .25,
               'source_count': 1, 'max_states': 2, 'max_alternatives': 3, 'max_emission_length': 2}
    data = {'context': context, 'records': ['xyxy', 'xx', 'yy', 'xy']}
    fit_spec = save(tmp_path/'fit.json', data)
    work, cache = [], {}
    def score(key):
        key = tuple(key)
        if key not in cache:
            values = [decode(source, key, r, .25) for r in data['records']]
            logs = [None if v.log_likelihood == -math.inf else v.log_likelihood for v in values]
            total = None if None in logs else math.fsum(logs)
            bits = 7+sum(map(len, key))
            cache[key] = {'units': list(key), 'record_log_likelihoods': logs, 'record_nodes': [v.reachable_nodes for v in values],
                          'record_edges': [v.edges for v in values], 'fit_log_likelihood': total, 'model_bits': bits,
                          'log_weight_unnormalized': None if total is None else total-bits*math.log(2)}
            work.append(cache[key])
        return cache[key]
    candidates = [('x', 'y'), ('y', 'x')]
    old_rows = sorted((score(k) for k in candidates), key=lambda r: -r['log_weight_unnormalized'])
    old_bank = {'bank': old_rows, 'best_index': 0}
    old_spec = save(tmp_path/'old-bank.gz', old_bank, True)
    legacy_spec = save(tmp_path/'legacy.gz', {'trace': [
        {'event': 'restart', 'restart': i, 'units': key, 'score': {'value': 1}, 'method': 'toy'}
        for i, key in enumerate(candidates)]}, True)
    parent_spec = save(tmp_path/'parent.json', {'fit': fit_spec, 'stage1_trace': legacy_spec,
                                               'stage1_accounting': {'scored_initializations': 2}})
    prior_spec = save(tmp_path/'prior.json', {'fit': fit_spec, 'best_units': old_rows[0]['units'], 'bank': old_spec})
    starts, initial = warm_starts(candidates, old_rows[0]['units'], score)
    search = search_tempered(lambda key: score(key)['log_weight_unnormalized'], starts, ('x', 'y'), config=cfg)
    neighbors = one_move_bank(search['best_units'], 'xy')
    for key in neighbors:
        score(key)
    reader = reader_union(old_bank, [cache[tuple(r['units'])] for r in search['bank']], [cache[key] for key in neighbors])
    packed = {'search': search, 'scores': work, 'warm': initial, 'warm_starts': starts, 'neighborhood_units': neighbors}
    search_spec = save(tmp_path/'search.gz', packed, True)
    reader_spec = save(tmp_path/'reader.gz', reader, True)
    warm_spec = save(tmp_path/'warm.json', {'parent': parent_spec, 'old_fit': prior_spec,
                                          'starts': starts, 'scores': initial, 'trace': legacy_spec})
    progress = [{'kind': 'score', 'value': r} for r in work]+[{'kind': 'event', 'value': e} for e in search['events']]
    progress_path = tmp_path/'progress.gz'
    progress_path.write_bytes(gzip.compress('\n'.join(json.dumps(r) for r in progress).encode(), mtime=0))
    best = reader['bank'][reader['best_index']]['log_weight_unnormalized']
    warm_best = max(r['log_weight_unnormalized'] for r in initial)
    save(out/'case-01.json', {'status': 'complete_fit_only', 'case': 'case-01', 'freeze': 'artificial',
        'fit': fit_spec, 'old_fit': prior_spec, 'old_bank': old_spec, 'parent': parent_spec,
        'warm': warm_spec, 'search': search_spec, 'reader_bank': reader_spec, 'progress': identity(progress_path),
        'config': asdict(cfg), 'unique_full_fit_scores_including_warm_and_neighborhood': len(work),
        'reader_summary': {k: v for k, v in reader.items() if k != 'bank'},
        'search_summary': {k: v for k, v in search.items() if k not in ('bank', 'events')},
        'fit_objective_gain_nats': best-old_rows[0]['log_weight_unnormalized'],
        'gain_vs_best_warm_nats': best-warm_best, 'search_gain_vs_best_warm_nats': search['best_score']-warm_best})
    monkeypatch.setattr(auditor, 'ROOT', tmp_path)
    monkeypatch.setattr(auditor, 'OUT', out)
    monkeypatch.setattr(auditor, 'artifact', identity)
    monkeypatch.setattr(auditor, 'save_new', save)
    monkeypatch.setattr(auditor, 'require_frozen', lambda *a: None)
    monkeypatch.setattr(auditor, 'admitted', lambda *a: ({'cases': {'case-01': {'fit': fit_spec}}}, None))
    monkeypatch.setattr(auditor, 'observed', lambda *a: data)
    monkeypatch.setattr(auditor, 'load_archive', lambda spec: json.loads(gzip.decompress((tmp_path/spec['path']).read_bytes())))
    monkeypatch.setattr(auditor, 'config_for', lambda *a: cfg)
    monkeypatch.setattr(auditor, 'limit_resources', lambda *a: None)
    monkeypatch.setattr(auditor, 'literal_source', lambda: {'alphabet': source.alphabet, 'order': 1, 'tau': 4.,
                                                          'counts': {k: dict(v) for k, v in source.counts.items()}})
    auditor.audit('case-01')
    result = json.loads((out/'case-01-audit.json').read_text())
    assert result['status'] == 'PASS' and result['reference_records'] >= 8 and result['maximum_delta'] < 1e-12
