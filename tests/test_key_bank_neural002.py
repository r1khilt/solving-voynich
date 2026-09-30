import io
import json
import math
import runpy

import pytest
import torch

from scripts import run_key_bank_neural002 as runner
from voynich.fixed_sequence_scoring import fixed_record_logps
from voynich.recurrent_latin_source import RecurrentSource


def test_configuration_restores_original_namespace_on_failure():
    keys = ('EXP', 'OUT', 'BULK', 'PATHS', 'sequence_scores')
    before = {k: getattr(runner.engine, k) for k in keys}
    with pytest.raises(RuntimeError):
        with runner.configured('sentinel'):
            assert runner.engine.EXP == 'KEY-BANK-NEURAL-002'
            assert runner.engine.sequence_scores == 'sentinel'
            assert runner.engine.OUT == runner.OUT and runner.engine.PATHS == runner.PATHS
            raise RuntimeError('deliberate')
    assert {k: getattr(runner.engine, k) for k in keys} == before


def test_gpu_adapter_preserves_scores_and_traces_every_inner_batch(monkeypatch):
    monkeypatch.setattr(runner, 'snapshot', lambda: {'driver_bytes': 10, 'tensor_bytes': 5})
    monkeypatch.setattr(runner, 'fixed_record_logps', lambda m, t, d, **kw: fixed_record_logps(m, t, 'cpu', **kw))
    torch.manual_seed(7451)
    model = RecurrentSource('ab', embedding=4, width=8, layers=1).double()
    texts = ['ab'*(i+1) for i in range(11)]
    stream = io.StringIO()
    scorer = runner.Scorer(stream, runner.engine.sequence_scores)
    values = scorer(model, texts, 'mps')
    expected = runner.engine.sequence_scores(model, texts, 'cpu')
    assert values == pytest.approx(expected, abs=1e-10, rel=0.)
    rows = [json.loads(r) for r in stream.getvalue().splitlines()]
    assert [r['records'] for r in rows] == [8, 11]
    assert all(r['batch_shape'] == [8, 270] for r in rows)
    assert scorer.records == 11 and scorer.characters == sum(map(len, texts))
    assert scorer.driver_peak == 10 and scorer.tensor_peak == 5
    old_trace = stream.getvalue()
    assert scorer(model, texts, 'cpu') == expected
    assert stream.getvalue() == old_trace and scorer.calls == 1
    with pytest.raises(ValueError):
        scorer(model, texts, 'cuda')


def test_gpu_adapter_saves_overlimit_sample_before_raising(monkeypatch):
    monkeypatch.setattr(runner, 'snapshot', lambda: {'driver_bytes': 2*1024**3+1, 'tensor_bytes': 5})
    monkeypatch.setattr(runner, 'fixed_record_logps', lambda m, t, d, **kw: fixed_record_logps(m, t, 'cpu', **kw))
    stream = io.StringIO()
    scorer = runner.Scorer(stream, runner.engine.sequence_scores)
    model = RecurrentSource('ab', embedding=4, width=8, layers=1)
    with pytest.raises(MemoryError):
        scorer(model, ['ab'], 'mps')
    assert json.loads(stream.getvalue())['driver_bytes'] == 2*1024**3+1
    assert scorer.records == 1 and scorer.characters == 0


@pytest.mark.parametrize('missing', [False, True])
def test_immutable_engine_staging_in_new_namespace(tmp_path, monkeypatch, missing):
    # Reuse the original artificial eight-case integration fixture. It enforces
    # no answer access during prediction and complete denominators after freeze.
    fixture = runpy.run_path(str(runner.ROOT/'tests/test_key_bank_neural001.py'))[
        'test_full_prediction_freeze_then_gold_evaluation_uses_all_eight_cases']
    original_namespace = runner.engine.EXP
    with runner.configured():
        with monkeypatch.context() as inner:
            fixture(tmp_path, inner, missing)
            value = json.loads((tmp_path/'results'/runner.EXP/'evaluation.json').read_text())
            assert value['experiment'] == runner.EXP
            assert not (tmp_path/'results'/'KEY-BANK-NEURAL-001').exists()
    assert runner.engine.EXP == original_namespace


def test_length_law_is_once_per_record(monkeypatch):
    def fake(_, texts, device, **kwargs):
        kwargs['after_batch']({'records': len(texts), 'batch_shape': [8, 270]})
        return [-3.]*len(texts)
    monkeypatch.setattr(runner, 'fixed_record_logps', fake)
    monkeypatch.setattr(runner, 'snapshot', lambda: {'driver_bytes': 0, 'tensor_bytes': 0})
    scorer = runner.Scorer(io.StringIO(), None)
    assert scorer(None, ['aaa'], 'mps') == [-3+3*math.log1p(-1/225)+math.log(1/225)]


def test_actual_gpu_adapter_matches_independent_cpu_reference():
    if not torch.backends.mps.is_available():
        pytest.skip('Separate device-access validation')
    torch.manual_seed(7489)
    model = RecurrentSource('ab', embedding=4, width=8, layers=1).to('mps')
    scorer = runner.Scorer(io.StringIO(), runner.engine.sequence_scores)
    texts = ['abba'*i for i in range(1, 10)]
    scores = scorer(model, texts, 'mps')
    reference = runner.engine.sequence_scores(model.to('cpu').double(), texts, 'cpu')
    assert scores == pytest.approx(reference, abs=1e-3, rel=0.)
    assert scorer.records == 9 and scorer.batches == 2
