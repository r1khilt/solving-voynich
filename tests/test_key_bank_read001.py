import copy
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_key_bank_read001 as runner
from scripts import evaluate_key_bank_read001 as evaluator
from scripts.evaluate_key_bank_read001 import arm_rows, truth_diagnosis
from scripts.run_blind_channel_dev004 import metrics
from voynich.shared_key_mixture import decode_shared_keys
from voynich.sparse_suffix_source import SuffixSource, decode


def bank(keys, weights):
    rows = [{'units': key, 'log_weight': math.log(w) if w else None} for key, w in zip(keys, weights)]
    return {'bank': rows, 'best_index': max(range(len(weights)), key=weights.__getitem__)}


def test_four_decisions_use_one_fixed_bank_and_same_source():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    keys = [('x', 'xx'), ('xx', 'x'), ('x', 'x'), ('y', 'y')]
    weights = [.4, .3, .3, 0.]
    data = bank(keys, weights)
    records = ['xxxx', 'xx']
    result = runner.read_case(source, records, keys[1], data, .2)
    assert result['status'] == 'complete'
    assert result['active_bank_indices'] == [0, 1, 2]
    assert result['checks']['keys'] == 3 and result['checks']['maximum_delta'] < 1e-12
    for arm, key in [('original', keys[1]), ('fit_selected', keys[0])]:
        assert result['points'][arm] == [decode(source, key, row, .2).to_dict() for row in records]
    joint_scores = [math.log(w) + sum(decode(source, key, r, .2).joint_log_probability for r in records)
                    for key, w in zip(keys, weights) if w]
    assert result['joint']['log_probability'] == pytest.approx(max(joint_scores), abs=1e-12)
    exact = decode_shared_keys(source, keys[:3], records, .2, log_weights=[math.log(w) for w in weights[:3]])
    mix = result['mixture']
    assert mix['log_likelihood'] == pytest.approx(exact.log_likelihood, abs=1e-12)
    assert mix['joint_log_probability'] <= exact.joint_log_probability + 1e-12
    assert exact.joint_log_probability <= max(mix['joint_log_probability'], mix['unseen_log_probability_upper_bound']) + 1e-12


def test_positive_tiny_weights_are_not_pruned_and_indices_map_back():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    data = {'bank': [{'units': ('z', 'z'), 'log_weight': None},
                     {'units': ('x', 'x'), 'log_weight': 0.},
                     {'units': ('y', 'y'), 'log_weight': -10000.}], 'best_index': 1}
    result = runner.read_case(source, ['y', 'y'], ('x', 'x'), data, .2)
    assert result['active_bank_indices'] == [1, 2]
    assert result['joint']['key_index'] == 2
    assert math.isfinite(result['mixture']['joint_log_probability'])
    assert result['points']['original'][0]['plaintext'] is None


def test_point_outputs_are_preserved_before_mixture_resource_failure(monkeypatch):
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    saved = []
    def fail(*args, **kwargs):
        raise RuntimeError('fixed cap')
    monkeypatch.setattr(runner, 'bounded_mixture', fail)
    with pytest.raises(RuntimeError, match='fixed cap'):
        runner.read_case(source, ['x'], ('x', 'y'), bank([('x', 'y')], [1.]), .2, save_points=saved.append)
    assert len(saved) == 1 and saved[0]['original'][0]['plaintext'] == 'a'
    assert saved[0]['fit_selected'][0]['plaintext'] == 'a'


def test_missing_fit_bank_retains_original_control_and_no_mixture():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    saved = []
    result = runner.read_case(source, ['x'], ('x', 'y'), None, .2, save_points=saved.append)
    assert result['status'] == 'fit_bank_unavailable' and result['mixture'] is None
    assert result['points']['original'][0]['plaintext'] == 'a' and len(saved) == 1


@pytest.mark.parametrize('arm', ['joint', 'mixture'])
def test_unsupported_completed_readings_are_not_dropped_from_conditional_metrics(arm):
    result = {'status': 'complete', 'joint': None, 'mixture': {'plaintexts': None}}
    rows, available = arm_rows(arm, {}, result, 2)
    assert available and metrics(rows, ['ab', 'aba'])['edits'] == 5
    assert metrics(rows, ['ab', 'aba'])['gold_characters'] == 5
    rows, available = arm_rows(arm, {}, None, 2)
    assert not available and metrics(rows, ['ab', 'aba'])['edits'] == 5


def test_incomplete_record_list_cannot_silently_shrink_denominator():
    with pytest.raises(ValueError, match='Incomplete'):
        arm_rows('original', {'original': [{'plaintext': 'a'}]}, None, 2)


def test_truth_diagnosis_distinguishes_bank_coverage_from_source_preference():
    source = SuffixSource('ab', 0, 1., {'': {'a': 90, 'b': 10}})
    missing = bank([('x', 'y')], [1.])
    reading = runner.read_case(source, ['x'], ('x', 'y'), missing, .2)
    value = truth_diagnosis(source, ['x'], ['b'], missing, reading, .2)
    assert value['status'] == 'true_tuple_outside_positive_weight_bank_support'
    ambiguous = bank([('x', 'x')], [1.])
    reading = runner.read_case(source, ['x'], ('x', 'x'), ambiguous, .2)
    value = truth_diagnosis(source, ['x'], ['b'], ambiguous, reading, .2)
    assert value['status'] == 'model_prefers_wrong_tuple' and value['supporting_keys'] == 1


def test_truth_diagnosis_detects_missed_candidate_and_false_bound_certificate():
    source = SuffixSource('ab', 0, 1., {'': {'a': 10, 'b': 90}})
    data = bank([('x', 'x')], [1.])
    reading = runner.read_case(source, ['x'], ('x', 'x'), data, .2)
    assert truth_diagnosis(source, ['x'], ['b'], data, reading, .2)['status'] == 'exact_true_tuple'
    fake = copy.deepcopy(reading)
    fake['mixture'].update(plaintexts=('a',), joint_log_probability=runner.path_score(source, ['a'], .2),
                           floating_bound_separated=False, unseen_log_probability_upper_bound=runner.path_score(source, ['b'], .2))
    assert truth_diagnosis(source, ['x'], ['b'], data, fake, .2)['status'] == 'known_better_true_tuple_missed'
    fake['mixture']['floating_bound_separated'] = True
    with pytest.raises(ValueError, match='separated'):
        truth_diagnosis(source, ['x'], ['b'], data, fake, .2)
    fake['mixture']['floating_bound_separated'] = False
    fake['mixture']['unseen_log_probability_upper_bound'] = fake['mixture']['joint_log_probability']
    with pytest.raises(ValueError, match='unseen'):
        truth_diagnosis(source, ['x'], ['b'], data, fake, .2)


def test_true_path_that_beats_exact_joint_map_is_validation_failure():
    source = SuffixSource('ab', 0, 1., {'': {'a': 10, 'b': 90}})
    data = bank([('x', 'x')], [1.])
    reading = runner.read_case(source, ['x'], ('x', 'x'), data, .2)
    reading['joint']['log_probability'] -= 1.
    with pytest.raises(ValueError, match='per-key MAP'):
        truth_diagnosis(source, ['x'], ['b'], data, reading, .2)


def test_complete_prediction_and_evaluation_file_flow_on_artificial_panel(tmp_path, monkeypatch):
    from voynich.compact_suffix_source import fit_compact
    out = tmp_path / 'results/KEY-BANK-READ-001'
    bulk = tmp_path / 'outputs/KEY-BANK-READ-001'
    events = []
    allow_gold = False
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
        if 'answer' in spec['path']:
            assert allow_gold and events[-1] in ('freeze', 'gold')
            events.append('gold')
        assert identity(tmp_path / spec['path']) == spec
        return json.loads((tmp_path / spec['path']).read_text())
    def load(spec):
        assert identity(tmp_path / spec['path']) == spec
        return json.loads(gzip.decompress((tmp_path / spec['path']).read_bytes()))
    for module in (runner, evaluator):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
        monkeypatch.setattr(module, 'OUT', out)
        monkeypatch.setattr(module, 'artifact', identity)
        monkeypatch.setattr(module, 'save_new', save)
        monkeypatch.setattr(module, 'checked_artifact', checked)
        monkeypatch.setattr(module, 'load_archive', load)
        monkeypatch.setattr(module, 'limit_resources', lambda *args: None)
        monkeypatch.setattr(module, 'require_frozen', lambda *args: events.append('freeze'))
    monkeypatch.setattr(runner, 'BULK', bulk)
    counts = fit_compact(['a'*90 + 'b'*10], 'ab', 0, 1)
    counts.save(tmp_path / 'counts.npz')
    save(tmp_path / 'results/LATIN-SOURCE-COMPACT-001/large.json',
         {'counts': identity(tmp_path / 'counts.npz'), 'selected': {'tau': 64.}})
    candidate_bank = bank([('x', 'xx'), ('xx', 'x')], [.6, .4])
    bank_spec = save(tmp_path / 'bank.json.gz', candidate_bank, compressed=True)
    admitted, manifest = {}, {'cases': {}}
    for name in runner.NAMES:
        positive = not name.endswith('-shuffle')
        transfer = save(tmp_path / f'data/{name}/transfer.json', {'records': ['x'*224, 'x'*224],
            'context': {'source_alphabet': ['a', 'b'], 'stop_probability': 1 / 225}})
        answer = save(tmp_path / f'data/{name}/answer.json', {'plaintext': {'transfer': ['a'*224, 'a'*224]}})
        manifest['cases'][name] = {'artifacts': {'transfer': transfer, 'answer': answer}}
        save(tmp_path / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json', {'units': ['x', 'xx']})
        admitted[name] = {'bank': bank_spec} if positive else None
    save(tmp_path / runner.MANIFEST, manifest)
    for module in (runner, evaluator):
        monkeypatch.setattr(module, 'admission', lambda freeze: (admitted, {'status': 'incomplete_failures_retained'}))
    old = {'archives': {'statistical-large': {}}}
    for name in runner.NAMES:
        runner.predict(name, 'artificial-prediction-freeze')
        points = load(json.loads((out / f'{name}-points.json').read_text())['readings'])
        old['archives']['statistical-large'][name] = save(tmp_path / f'old/{name}.json.gz',
                                                         {'learned': {'transfer': points['original']}}, compressed=True)
    assert 'gold' not in events
    save(tmp_path / 'results/KEY-SOURCE-DIAG-001/evaluation.json', old)
    save(out / 'campaign-started.json', {'freeze': 'artificial'})
    save(out / 'campaign.json', {'status': 'complete', 'results': [{'case': n, 'returncode': 0} for n in runner.NAMES]})
    allow_gold = True
    evaluator.evaluate('artificial-post-prediction-freeze')
    result = json.loads((out / 'evaluation.json').read_text())
    assert events[0] == 'freeze' and events.count('gold') == 8
    assert result['independent_record_edit_checks'] == 64 and result['baseline_records_reproduced'] == 32
    assert all(row['gold_characters'] == 3584 for row in result['totals'].values())
    assert all(row['cases'] == 8 for row in result['completed_case_metrics_conditional'].values())
    assert result['cases']['B-key1-shuffle']['inference_status'] == 'fit_bank_unavailable'
