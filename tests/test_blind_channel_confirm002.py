"""Artificial-only end-to-end staging, independent audits, and retained failures."""
import copy
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from scripts import audit_blind_channel_confirm002 as auditor
from scripts import confirm002_common as common
from scripts import evaluate_blind_channel_confirm002 as evaluator
from scripts import predict_blind_channel_confirm002 as predictor
from scripts import run_blind_channel_confirm002 as fitter
from scripts.run_blind_channel_dev001 import fit_glyph_baseline
from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import MarkovSource, contexts_for
from voynich.native_suffix_marginal import marginal_python


class PythonScorer:
    def __init__(self, source, *_):
        self.source, self.alphabet = source, source.alphabet

    def score(self, units, record, rho, **kwargs):
        return marginal_python(self.source, units, record, rho, **kwargs)


def initial_sources():
    alphabet = ('a', 'b')
    probabilities = {h: {'a': .75, 'b': .25} for h in contexts_for(alphabet, 3)}
    first = SourceModel(alphabet, 1, {h: probabilities[h] for h in ('', 'a', 'b')})
    third = MarkovSource(alphabet, 3, probabilities)
    return first, third, {'order1': first.to_dict(), 'order3': third.to_dict()}


def setup_files(tmp_path, monkeypatch):
    out, bulk = tmp_path/'results'/common.EXP, tmp_path/'outputs'/common.EXP
    opened, allowed = [], [False]
    def identity(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode()
        with path.open('xb') as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return identity(path)
    def checked(spec):
        opened.append(spec['path'])
        if 'answer' in spec['path']:
            assert allowed[0], 'Gold opened before audited prediction freeze'
        assert identity(tmp_path/spec['path']) == spec
        return json.loads((tmp_path/spec['path']).read_text())
    def archive(spec):
        assert identity(tmp_path/spec['path']) == spec
        return json.loads(gzip.decompress((tmp_path/spec['path']).read_bytes()))
    def verify(spec):
        assert identity(tmp_path/spec['path']) == spec
    compact = fit_compact(['a']*7+['b']*3, 'ab', 12, 4)
    path = tmp_path/'counts.npz'
    compact.save(path)
    source = DenseSuffixAdapter(compact, 64.)
    raw = {'alphabet': ('a', 'b'), 'order': 12, 'tau': 64., 'counts': {'': {'a': 7, 'b': 3}}}
    save(tmp_path/'results/LATIN-SOURCE-COMPACT-001/large.json', {'counts': identity(path), 'selected': {'tau': 64.}})
    save(tmp_path/'results/NATIVE-SUFFIX-001/result.json', {'build': {}})
    save(tmp_path/'scripts/audit_blind_channel_confirm002.py', {'artificial': True})
    for module in (fitter, predictor, auditor, evaluator, common):
        replacements = {'ROOT': tmp_path, 'OUT': out, 'BULK': bulk, 'artifact': identity, 'save_new': save,
                        'checked_artifact': checked, 'load_archive': archive, 'verify': verify,
                        'load_source': lambda: source, 'sources': initial_sources,
                        'literal_source': lambda: raw, 'limit_resources': lambda *_: None,
                        'NativeMarginal': PythonScorer}
        for key, value in replacements.items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    return SimpleNamespace(out=out, bulk=bulk, save=save, checked=checked, identity=identity,
                           archive=archive, opened=opened, allowed=allowed, raw=raw, source=source)


def test_real_three_stage_fit_and_independent_audit_use_only_fit_data(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    name = common.NAMES[0]
    context = CodingContext(('a', 'b'), ('X', 'Y'), 32, 2, 2, 3, stop_probability=1/225)
    spec = f.save(tmp_path/'fit.json', {'records': ['XXX', 'XY', 'X', 'Y'], 'context': asdict(context)})
    panel = {'source_freeze': 'source', 'cases': {name: {'fit': spec, 'search_seed': 73101}}}
    for module in (fitter, auditor):
        monkeypatch.setattr(module, 'fit_inputs', lambda *_: (panel, {'build': {}}))
        monkeypatch.setattr(module, 'observed', lambda value, count: f.checked(value))
    monkeypatch.setattr(fitter, 'PATHS', ['scripts/audit_blind_channel_confirm002.py'])
    monkeypatch.setattr(auditor, 'PATHS', ['scripts/audit_blind_channel_confirm002.py'])
    fitter.fit(name, 'panel')
    auditor.audit_fit(name)
    result = json.loads((f.out/f'{name}-fit.json').read_text())
    audit = json.loads((f.out/f'{name}-fit-audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['initial_selected_record_replays'] == 8
    assert audit['candidates_accounted'] == result['bank_size'] > 10
    assert all('answer' not in p and 'transfer' not in p for p in f.opened)
    with pytest.raises(FileExistsError):
        fitter.fit(name, 'panel')


@pytest.mark.parametrize('late_failure', [False, True])
def test_32_case_prediction_audit_evaluation_flow_without_gold(tmp_path, monkeypatch, late_failure):
    f = setup_files(tmp_path, monkeypatch)
    panel, admitted, processes = {'cases': {}}, {}, []
    fit_campaign = {'status': 'complete'}
    context = {'source_alphabet': ['a', 'b'], 'stop_probability': 1/225}
    bank = {'bank': [{'units': ['X', 'Y'], 'log_weight': -math.log(2)},
                     {'units': ['Y', 'X'], 'log_weight': -math.log(2)}],
            'best_index': 0, 'best_units': ['X', 'Y']}
    for i, name in enumerate(common.NAMES):
        spec = f.save(tmp_path/f'{name}-transfer.json', {'records': ['X'*224]*2, 'context': context})
        positive = i < 16
        answer = f.save(tmp_path/f'{name}-answer.json', {'positive': positive, 'pair_index': i % 16,
            'plaintext': {'transfer': ['a'*224]*2, 'fit': ['a'*224]*4} if positive else None, 'units': ['X', 'Y']})
        panel['cases'][name] = {'transfer': spec, 'answer': answer, 'positive': positive, 'pair_index': i % 16}
        parent = f.save(tmp_path/f'{name}-parent.json', {'units': ['X', 'Y'], 'frequency_units': ['Y', 'X'],
                        'baseline': fit_glyph_baseline(['XXY', 'YYX'], ('X', 'Y'))})
        admitted[name] = {'parent': parent, 'bank': f.save(tmp_path/f'{name}-bank.gz', bank, compressed=True)}
    for module in (predictor, auditor, evaluator):
        monkeypatch.setattr(module, 'fit_status', lambda *_: (panel, {'build': {}}, admitted, fit_campaign))
    original_read = predictor.read_case
    def fail_after_points(*args, **kwargs):
        keep = kwargs['save_points']
        def save_then_fail(value):
            keep(value)
            raise RuntimeError('Artificial fixed cap')
        kwargs['save_points'] = save_then_fail
        return original_read(*args, **kwargs)
    for i, name in enumerate(common.NAMES):
        fail = late_failure and i == 0
        monkeypatch.setattr(predictor, 'read_case', fail_after_points if fail else original_read)
        if fail:
            with pytest.raises(RuntimeError, match='fixed cap'):
                predictor.predict(name, 'fit-freeze')
        else:
            predictor.predict(name, 'fit-freeze')
        processes.append({'case': name, 'returncode': 1 if fail else 0})
    assert not any('answer' in p for p in f.opened)
    f.save(f.out/'campaign.json', {'freeze': 'fit-freeze', 'results': processes})
    f.save(f.out/'fit-campaign.json', fit_campaign)
    auditor.audit_predictions()
    assert not any('answer' in p for p in f.opened)
    with pytest.raises(FileExistsError):
        predictor.predict(common.NAMES[0], 'fit-freeze')
    def freeze(_commit, paths):
        assert (f.out/'audit.json').exists()
        assert any(p.endswith('/audit.json') for p in paths)
        assert not any('answer' in p for p in f.opened)
        f.allowed[0] = True
    monkeypatch.setattr(evaluator, 'require_frozen', freeze)
    monkeypatch.setattr(evaluator, 'observed', lambda spec, count: f.checked(spec))
    evaluator.evaluate('prediction-freeze')
    result = json.loads((f.out/'evaluation.json').read_text())
    assert result['totals']['mixture']['gold_characters'] == 7168
    assert result['totals']['mixture']['edits'] == (448 if late_failure else 0)
    assert result['completed_case_metrics_conditional']['mixture']['cases'] == (15 if late_failure else 16)
    assert result['independent_record_edit_checks'] == 192 and result['independent_oracle_records'] == 32
    assert result['recovery_gate'] == ('FAIL' if late_failure else 'PASS')
    assert not result['screen_clauses']['no_shuffle_beats_frozen_iid']
    assert sum('answer' in p for p in f.opened) == 16


def gate_fixture():
    reports = {}
    for i, name in enumerate(common.NAMES):
        reports[name] = {'positive': i < 16, 'fit_complete': True, 'prediction_complete': True,
                         'evidence': {'gain_bits_vs_iid': 1 if i < 16 else -1}}
        if i < 16:
            reports[name]['arms'] = {arm: {'edits': 1 if arm == 'mixture' else 100 if arm == 'frequency_order3' else 0,
                                          'gold_characters': 448, 'supported_records': 2} for arm in evaluator.ALL_ARMS}
    return reports


@pytest.mark.parametrize('fault', ['null_fit_failure', 'null_prediction_failure', 'noise_false_positive',
                                  'positive_not_above_iid', 'missing_evidence', 'individual_cer',
                                  'mean_cer', 'oracle_gap', 'frequency_exact', 'unsupported'])
def test_fixed_gates_fail_for_all_failure_types(fault):
    reports = gate_fixture()
    assert all(evaluator.decisions(reports)[0].values())
    assert all(evaluator.decisions(reports)[1].values())
    positive, negative = reports[common.NAMES[0]], reports[common.NAMES[16]]
    if fault == 'null_fit_failure':
        negative['fit_complete'] = False
    elif fault == 'null_prediction_failure':
        negative['prediction_complete'] = False
    elif fault == 'noise_false_positive':
        negative['evidence']['gain_bits_vs_iid'] = .001
    elif fault == 'positive_not_above_iid':
        positive['evidence']['gain_bits_vs_iid'] = 0
    elif fault == 'missing_evidence':
        negative['evidence'] = None
    elif fault == 'individual_cer':
        positive['arms']['mixture']['edits'] = 23
    elif fault == 'mean_cer':
        for row in list(reports.values())[:16]:
            row['arms']['mixture']['edits'] = 9
            row['arms']['oracle_large']['edits'] = 9
    elif fault == 'oracle_gap':
        positive['arms']['mixture']['edits'] = 9
    elif fault == 'frequency_exact':
        positive['arms']['frequency_order3']['edits'] = 0
    else:
        positive['arms']['mixture']['supported_records'] = 1
    recovery, screen = evaluator.decisions(reports)
    assert not all(recovery.values()) or not all(screen.values())


def test_oracle_and_point_audits_catch_score_and_encoding_corruption(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    rows, delta = evaluator.oracle_rows(f.source, f.raw, ['X', 'YY'], ['XXX', 'XYY'], .2)
    assert delta < 1e-12
    parent, bank = {'units': ['X', 'YY']}, {'best_units': ['X', 'YY']}
    points = {'original': rows, 'fit_selected': copy.deepcopy(rows)}
    assert auditor.audit_points(f.raw, ['XXX', 'XYY'], parent, bank, points, .2) == 4
    points['fit_selected'][0]['joint_log_probability'] += .01
    with pytest.raises(ValueError):
        auditor.audit_points(f.raw, ['XXX', 'XYY'], parent, bank, points, .2)


def test_campaign_retains_fit_audit_timeout_and_launch_failures(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    names = common.NAMES[:4]
    monkeypatch.setattr(fitter, 'NAMES', names)
    monkeypatch.setattr(fitter, 'fit_inputs', lambda *_: ({'source_freeze': 'source'}, {}))
    calls = []
    def run(command, **kwargs):
        name = command[command.index('--case')+1]
        audit = 'audit_blind_channel_confirm002.py' in command[2]
        calls.append((name, audit))
        if name == names[1]:
            return SimpleNamespace(returncode=1)
        if name == names[2] and audit:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if name == names[3]:
            raise OSError('Artificial unavailable executable')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(fitter.subprocess, 'run', run)
    fitter.campaign('panel')
    value = json.loads((f.out/'fit-campaign.json').read_text())
    assert [row['case'] for row in value['results']] == list(names)
    assert value['status'] == 'failures_retained'
    assert (names[1], True) not in calls and (names[3], True) not in calls
    assert value['results'][2]['audit_status'] == 'outer_timeout'
    assert value['results'][3]['fit_status'] == 'launch_failure'
    with pytest.raises(FileExistsError):
        fitter.campaign('panel')


def test_fit_panel_rejects_transfer_gold_labels_or_changed_search_seeds(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    panel = {'experiment': common.EXP, 'source_freeze': 'source', 'status': 'prepared_fit_only',
             'cases': {name: {'fit': {'path': 'opaque.json'}, 'search_seed': 73101+257*i}
                       for i, name in enumerate(common.NAMES)}}
    monkeypatch.setattr(common, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(common, 'source_admission', lambda *_: {})
    path = tmp_path/common.FIT_PANEL
    f.save(path, panel)
    assert common.fit_inputs('panel')[0] == panel
    for key in ('positive', 'answer', 'transfer', 'key_seed'):
        bad = copy.deepcopy(panel)
        bad['cases'][common.NAMES[0]][key] = 'forbidden'
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='boundary'):
            common.fit_inputs('panel')
    panel['cases'][common.NAMES[0]]['search_seed'] += 1
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match='seed'):
        common.fit_inputs('panel')


def test_process_failure_does_not_create_or_admit_a_partial_normalized_bank(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    name = common.NAMES[0]
    monkeypatch.setattr(fitter, 'fit_inputs', lambda *_: ({'source_freeze': 'source',
        'cases': {name: {'fit': {'path': 'fit.json'}, 'search_seed': 73101}}}, {}))
    def fail(*_):
        raise RuntimeError('Artificial hard fit failure')
    monkeypatch.setattr(fitter, 'observed', fail)
    with pytest.raises(RuntimeError, match='hard fit'):
        fitter.fit(name, 'panel')
    assert (f.out/f'{name}-fit-failure.json').exists()
    assert not (f.out/f'{name}-fit.json').exists()
    with pytest.raises(FileExistsError):
        fitter.fit(name, 'panel')


def test_fit_status_requires_complete_frozen_accounting_and_matching_audits(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    panel, processes = {'source_freeze': 'source', 'cases': {}}, []
    for name in common.NAMES:
        fit = {'path': f'{name}-ciphertext.json'}
        panel['cases'][name] = {'fit': fit}
        row = {'case': name, 'fit_returncode': 0, 'audit_returncode': 0}
        processes.append(row)
        result = {'status': 'complete_fit_only', 'case': name, 'fit': fit, 'data_freeze': 'panel',
                  'source_freeze': 'source', 'bank_size': 1}
        spec = f.save(f.out/f'{name}-fit.json', result)
        f.save(f.out/f'{name}-fit-audit.json', {'status': 'PASS', 'result': spec, 'candidates_accounted': 1,
               'auditor': f.identity(tmp_path/'scripts/audit_blind_channel_confirm002.py')})
        f.save(f.out/f'{name}-fit-process.json', row)
    campaign = {'source_freeze': 'source', 'data_freeze': 'panel', 'results': processes}
    path = f.out/'fit-campaign.json'
    f.save(path, campaign)
    monkeypatch.setattr(common, 'panel_inputs', lambda *_: (panel, {}))
    frozen = []
    monkeypatch.setattr(common, 'require_frozen', lambda commit, paths: frozen.extend(paths))
    assert all(common.fit_status('fits')[2].values())
    assert any(p.endswith('case-32-fit-audit.json') for p in frozen)
    processes[0]['audit_returncode'] = 1
    path.write_text(json.dumps(campaign))
    (f.out/f'{common.NAMES[0]}-fit-process.json').write_text(json.dumps(processes[0]))
    assert common.fit_status('fits')[2][common.NAMES[0]] is None
    processes.pop()
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match='terminal'):
        common.fit_status('fits')
