import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_key_bank_read002 as runner
from scripts import evaluate_key_bank_read002 as evaluator
from scripts.audit_key_bank_read002 import audit_evidence
from scripts.benchmark_shared_key002 import serializable
from voynich.native_suffix_marginal import marginal_python
from voynich.shared_key_mixture import decode_shared_keys
from voynich.sparse_suffix_source import SuffixSource


class PythonScorer:
    def __init__(self, source, *_):
        self.source = source

    def score(self, units, record, rho, **kwargs):
        return marginal_python(self.source, units, record, rho, **kwargs)


def fixture():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    bank = {'bank': [{'units': k, 'log_weight': w} for k, w in
                    [(('x', 'xx'), -math.log(3)), (('xx', 'x'), -math.log(3)),
                     (('x', 'x'), -math.log(3)), (('z', 'z'), None)]], 'best_index': 0}
    return source, bank


def test_predictive_evidence_equals_exhaustive_joint_consistency_decoder():
    source, bank = fixture()
    records = ['xxxx', 'xx']
    actual = runner.evidence_case(PythonScorer(source), records, bank, .2)
    expected = decode_shared_keys(source, [r['units'] for r in bank['bank'][:3]], records, .2,
                                  log_weights=[r['log_weight'] for r in bank['bank'][:3]])
    assert actual['log_likelihood'] == pytest.approx(expected.log_likelihood, abs=1e-12)
    assert audit_evidence(bank, serializable(actual))['active_keys'] == 3


def test_cannot_change_keys_between_records_and_tiny_weights_survive():
    source, _ = fixture()
    bank = {'bank': [{'units': ('x', 'x'), 'log_weight': 0.},
                     {'units': ('y', 'y'), 'log_weight': -10000.}]}
    impossible = runner.evidence_case(PythonScorer(source), ['x', 'y'], bank, .2)
    assert impossible['log_likelihood'] == -math.inf and impossible['supported_keys'] == 0
    possible = runner.evidence_case(PythonScorer(source), ['y', 'y'], bank, .2)
    assert possible['log_likelihood'] < -10000 and math.isfinite(possible['log_likelihood'])
    assert possible['active_bank_indices'] == [0, 1]


@pytest.mark.parametrize('fault', ['inventory', 'record', 'total', 'count', 'weights'])
def test_corruption_cannot_pass_evidence_audit(fault):
    source, bank = fixture()
    value = serializable(runner.evidence_case(PythonScorer(source), ['xx', 'x'], bank, .2))
    if fault == 'inventory':
        value['active_bank_indices'].pop()
    elif fault == 'record':
        value['per_key'][0]['record_log_likelihoods'][0] += 1
    elif fault == 'total':
        value['log_likelihood'] += 1
    elif fault == 'count':
        value['supported_keys'] -= 1
    else:
        bank['bank'][0]['log_weight'] += 1
    with pytest.raises(ValueError):
        audit_evidence(bank, value)


def test_duplicates_and_unnormalized_weights_rejected():
    source, bank = fixture()
    bank['bank'][1]['units'] = bank['bank'][0]['units']
    with pytest.raises(ValueError, match='Duplicate'):
        runner.evidence_case(PythonScorer(source), ['x'], bank, .2)
    source, bank = fixture()
    bank['bank'][0]['log_weight'] = 1.
    with pytest.raises(ValueError):
        runner.evidence_case(PythonScorer(source), ['x'], bank, .2)


def test_preserve_evidence_and_points_before_map_failure(tmp_path, monkeypatch):
    # Full file-flow fixture below injects a late failure on one positive case.
    run_flow(tmp_path, monkeypatch, fail_case='B-key1')


def test_all_case_file_flow_excludes_null_gold_and_obeys_prediction_freeze(tmp_path, monkeypatch):
    run_flow(tmp_path, monkeypatch)


def run_flow(tmp_path, monkeypatch, fail_case=None):
    out, bulk = tmp_path / 'results/KEY-BANK-READ-002', tmp_path / 'outputs/KEY-BANK-READ-002'
    events, allowed = [], [False]
    def identity(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode()
        with path.open('xb') as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return identity(path)
    def load(spec):
        assert identity(tmp_path/spec['path']) == spec
        return json.loads(gzip.decompress((tmp_path/spec['path']).read_bytes()))
    def checked(spec):
        if 'answer' in spec['path']:
            assert allowed[0] and events[-1] in ('freeze', 'gold')
            assert 'shuffle' not in spec['path']
            events.append('gold')
        assert identity(tmp_path/spec['path']) == spec
        return json.loads((tmp_path/spec['path']).read_text())
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    for module in (runner, evaluator):
        for key, value in {'ROOT': tmp_path, 'OUT': out, 'artifact': identity, 'save_new': save,
                           'load_archive': load, 'checked_artifact': checked,
                           'load_source': lambda: source, 'limit_resources': lambda *_: None,
                           'require_frozen': lambda *_: events.append('freeze')}.items():
            monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(runner, 'BULK', bulk)
    monkeypatch.setattr(runner, 'NativeMarginal', PythonScorer)
    manifest, admitted = {'cases': {}}, {}
    bank = {'bank': [{'units': ('x', 'y'), 'log_weight': -math.log(2)},
                     {'units': ('y', 'x'), 'log_weight': -math.log(2)}], 'best_index': 0}
    for name in runner.NAMES:
        transfer = save(tmp_path / f'{name}-transfer.json', {'records': ['x'*224]*2,
            'context': {'source_alphabet': list('ab'), 'stop_probability': 1/225}})
        answer = save(tmp_path / f'{name}-answer.json', {'plaintext': {'transfer': ['a'*224]*2}})
        manifest['cases'][name] = {'artifacts': {'transfer': transfer, 'answer': answer}}
        save(tmp_path / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json',
             {'units': ['x', 'y'], 'baseline': {'glyph_alphabet': ['x', 'y'], 'counts': [1, 1],
               'denominator': 2, 'stop_count': 1, 'stop_denominator': 225}})
        admitted[name] = {'bank': save(tmp_path / f'{name}-bank.gz', bank, compressed=True)}
    save(tmp_path / runner.MANIFEST, manifest)
    for module in (runner, evaluator):
        monkeypatch.setattr(module, 'admission', lambda *_: (admitted, {'build': {}}))
    previous_reader = runner.read_case
    processes = []
    for name in runner.NAMES:
        def failing_reader(*args, **kwargs):
            # Delegate point construction, then inject failure at its save boundary.
            callback = kwargs['save_points']
            def points_then_fail(points):
                callback(points)
                raise RuntimeError('fixed MAP cap')
            kwargs['save_points'] = points_then_fail
            return previous_reader(*args, **kwargs)
        monkeypatch.setattr(runner, 'read_case', failing_reader if name == fail_case else previous_reader)
        if name == fail_case:
            with pytest.raises(RuntimeError, match='fixed MAP cap'):
                runner.predict(name, 'source-freeze')
            assert (out / f'{name}-evidence.json').exists() and (out / f'{name}-points.json').exists()
        else:
            runner.predict(name, 'source-freeze')
        processes.append({'case': name, 'returncode': 1 if name == fail_case else 0})
        if not name.endswith('-shuffle'):
            saved = json.loads((out / f'{name}-points.json').read_text())
            save(tmp_path / f'results/KEY-BANK-READ-001/{name}-points.json', saved)
        else:
            assert not (out / f'{name}-points.json').exists()
            assert json.loads((out / f'{name}.json').read_text())['status'] == 'complete_evidence_only_by_design'
    assert 'gold' not in events
    save(tmp_path / 'results/KEY-BANK-READ-001/evaluation.json', {'totals': {'mixture': {'edits': 45}}})
    save(out / 'campaign.json', {'freeze': 'source-freeze', 'results': processes})
    auditor = save(tmp_path / 'scripts/audit_key_bank_read002.py', {'fixture': True})
    save(out / 'audit.json', {'status': 'PASS', 'campaign': identity(out/'campaign.json'), 'auditor': auditor})
    allowed[0] = True
    evaluator.evaluate('prediction-freeze')
    result = json.loads((out / 'evaluation.json').read_text())
    assert events.count('gold') == 8 and result['independent_record_edit_checks'] == 64
    assert result['totals']['mixture']['gold_characters'] == 3584
    assert result['totals']['mixture']['edits'] == (448 if fail_case else 0)
    assert result['reading_development_gate'] == ('FAIL' if fail_case else 'PASS')
    assert result['completed_case_metrics_conditional']['mixture']['cases'] == (7 if fail_case else 8)
    assert not result['screen_clauses']['no_shuffle_beats_frozen_iid']
