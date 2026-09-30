import gzip
import hashlib
import json
import math

import pytest
import torch

from scripts import run_key_bank_neural001 as runner
from scripts.benchmark_shared_key002 import serializable
from scripts.run_key_bank_read001 import read_case
from voynich.candidate_rerank import prepare_candidates, rank_candidates
from voynich.recurrent_latin_source import RecurrentSource
from voynich.sparse_suffix_source import SuffixSource


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_padded_scores_stepwise_double_and_complete_histories(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Explicit Mac device-access check run separately')
    torch.manual_seed(8161)
    model = RecurrentSource('ab', embedding=4, width=8, layers=1).to(device).eval()
    texts = ['a', 'ba', 'abba', 'a'*513]
    scores = runner.sequence_scores(model, texts, device)
    individual = [runner.sequence_scores(model, [text], device)[0] for text in texts]
    assert scores == pytest.approx(individual, abs=1e-4, rel=0.)
    prepared = {'texts': texts, 'candidates': [
        {'text_indices': [0, 1], 'key_log_mass': -.1}, {'text_indices': [2, 3], 'key_log_mass': -.2}]}
    audit = runner.numerical_audit(model, prepared, scores, rank_candidates(prepared, scores))
    assert audit['maximum_mps_float64_delta'] < 1e-3
    assert audit['maximum_stepwise_delta'] < 1e-7


@pytest.mark.parametrize('missing', [False, True])
def test_full_prediction_freeze_then_gold_evaluation_uses_all_eight_cases(tmp_path, monkeypatch, missing):
    out, bulk = tmp_path/'results'/runner.EXP, tmp_path/'outputs'/runner.EXP
    gold_allowed, gold_reads = [False], []
    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode()
        with path.open('xb') as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return artifact(path)
    def load(spec):
        assert artifact(tmp_path/spec['path']) == spec
        return json.loads(gzip.decompress((tmp_path/spec['path']).read_bytes()))
    def checked(spec):
        if 'answer' in spec['path']:
            assert gold_allowed[0]
            gold_reads.append(spec['path'])
        assert artifact(tmp_path/spec['path']) == spec
        return json.loads((tmp_path/spec['path']).read_text())
    for key, value in {'ROOT': tmp_path, 'OUT': out, 'BULK': bulk, 'ALPHABET': 'ab', 'save_new': save,
                       'artifact': artifact, 'load_archive': load, 'checked_artifact': checked,
                       'limit_resources': lambda *_: None}.items():
        monkeypatch.setattr(runner, key, value)
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    bank = {'bank': [{'units': ['X', 'Y'], 'log_weight': -math.log(2)},
                     {'units': ['Y', 'X'], 'log_weight': -math.log(2)}], 'best_index': 0}
    bank_spec = save(tmp_path/'bank.gz', bank, compressed=True)
    headers, manifest = {}, {'cases': {}}
    for name in runner.NAMES:
        transfer = save(tmp_path/f'{name}-transfer.json', {'records': ['X'*224]*2})
        answer = save(tmp_path/f'{name}-answer.json', {'plaintext': {'transfer': ['a'*224]*2}})
        reading = serializable(read_case(source, ['X'*224]*2, ['X', 'Y'], bank, 1/225))
        headers[name] = {'bank': bank_spec, 'transfer': transfer,
                         'readings': save(tmp_path/f'{name}-reading.gz', reading, compressed=True)}
        manifest['cases'][name] = {'artifacts': {'answer': answer}}
        save(tmp_path/f'results/KEY-BANK-READ-002/{name}.json', headers[name])
    save(tmp_path/runner.MANIFEST, manifest)
    save(tmp_path/'results/KEY-BANK-READ-002/evaluation.json', {'totals': {'mixture': {'edits': 43}}})
    monkeypatch.setattr(runner, 'inputs', lambda *_: (headers, {arm: {} for arm in runner.ARMS}))
    torch.manual_seed(8181)
    model = RecurrentSource('ab', embedding=4, width=8, layers=1).eval()
    monkeypatch.setattr(runner, 'load_neural', lambda *_: model)
    original = runner.sequence_scores
    monkeypatch.setattr(runner, 'sequence_scores', lambda model, texts, device: original(model, texts, 'cpu'))
    monkeypatch.setattr(runner.torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(runner.torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(runner.torch.mps, 'empty_cache', lambda: None)
    runner.predict('source-freeze')
    assert not gold_reads
    predictions = json.loads((out/'predictions.json').read_text())
    assert len(predictions['cases']) == 8
    assert all(set(row['predictions']) == set(runner.ARMS) for row in predictions['cases'].values())
    with pytest.raises(FileExistsError):
        runner.predict('source-freeze')
    if missing:
        (out/'predictions.json').unlink()
        (out/f'{runner.NAMES[-1]}-{runner.ARMS[-1]}.json').unlink()
        save(out/'prediction-failure.json', {'type': 'artificial failure after partial predictions'})
    def freeze(commit, paths):
        expected = '/prediction-failure.json' if missing else '/predictions.json'
        assert commit == 'prediction-freeze' and any(p.endswith(expected) for p in paths)
        gold_allowed[0] = True
    monkeypatch.setattr(runner, 'require_frozen', freeze)
    runner.evaluate('prediction-freeze')
    value = json.loads((out/'evaluation.json').read_text())
    assert len(gold_reads) == 8 and value['independent_edit_checks'] == 32
    assert all(row['gold_characters'] == 3584 for row in value['totals'].values())
    if missing:
        assert not value['development_clauses']['all_sixteen_case_model_predictions_complete_audited']
        lost = value['cases'][runner.NAMES[-1]][runner.ARMS[-1]]
        assert not lost['available'] and lost['record_edits'] == [224, 224]


def test_literal_support_replay_rejects_tampered_mass(monkeypatch):
    monkeypatch.setattr(runner, 'ALPHABET', 'ab')
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    bank = {'bank': [{'units': ['X', 'XX'], 'log_weight': 0.}], 'best_index': 0}
    reading = serializable(read_case(source, ['XXXX', 'XX'], ['X', 'XX'], bank, .2))
    prepared = prepare_candidates('ab', ['XXXX', 'XX'], bank, reading)
    assert runner.literal_support_audit(prepared, bank, ['XXXX', 'XX'], [0])['maximum_delta'] < 1e-12
    prepared['candidates'][0]['key_log_mass'] += 1
    with pytest.raises(ValueError):
        runner.literal_support_audit(prepared, bank, ['XXXX', 'XX'], [0])
