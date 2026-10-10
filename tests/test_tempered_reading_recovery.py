"""Fresh allocation, cipher-only initialization, cold selection and sealed recovery schemas."""
import gzip
import hashlib
import json
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from scripts import run_tempered_reading_recovery001 as run
from scripts.audit_tempered_reading_recovery001 import (audit, independent_metrics,
    independent_recovery_replay, verify_cold_selection)
from tests.test_tempered_reading_admission import source_fixture, initial_fixture
from voynich.joint_key_training import EpisodeSampler, dictionary_code
from voynich.recurrent_latin_source import ALPHABET
from voynich.tempered_cold_selection import receipt_target, select_cold
from voynich.tempered_reading_trace import replica_trace


def test_cold_selection_separate_replay_and_degree_one_empty_exchange():
    source = source_fixture()
    sampler, initial = initial_fixture(source)
    for degrees, sweeps in (((1,), 8), ((1, 2, 4, 16), 2)):
        trace = replica_trace(sampler, initial.actions, seed=96221, sweeps=sweeps, degrees=degrees)
        assert independent_recovery_replay(source, sampler, trace, degrees=degrees, sweeps=sweeps)['local_attempts'] == 8
        for receipt in [trace['initial'], *(r['retained'][0] for r in trace['trace'])]:
            path = sampler.path(forced_actions=tuple(receipt['actions']))
            assert F(*receipt_target(receipt)) == F(*sampler.target_integers(path))
        trace['selection'] = select_cold(trace)
        assert verify_cold_selection(source, sampler, trace) == trace['selection']['reading']
        if degrees == (1,):
            assert all(r['swaps'] == [] for r in trace['trace'])
        trace['selection']['selected_sweep'] = 12345
        with pytest.raises(AssertionError):
            verify_cold_selection(source, sampler, trace)


def test_fresh_allocation_forbids_old_and_duplicate_keys(monkeypatch):
    texts = ['abacabadabacaba'*40]
    sampler = EpisodeSampler(texts)
    old = sampler.sample(__import__('numpy').random.default_rng(1))
    manifest = {'old_forbidden_raw': [dictionary_code(old[2]['raw_indices'])],
                'old_forbidden_canonical': [dictionary_code(old[1])], 'validation_source': {'artificial': True}}
    monkeypatch.setattr(run, 'POSITIVE', 4)
    monkeypatch.setattr(run, 'CONTROL_COUNT', 2)
    monkeypatch.setattr(run, 'CASES', 9)
    monkeypatch.setattr(run.admitted.prior.training, 'load_prepared', lambda: (manifest, [], texts, [old]))
    monkeypatch.setattr(run.training, 'artifact', lambda _: {'artificial': True})
    _, episodes, allocated, provenance = run.allocate()
    assert len(episodes) == 4 and len(allocated) == 9
    assert len({dictionary_code(e[1]) for e in episodes}) == 4
    assert all(e[1] != old[1] for e in episodes)
    assert provenance['no_new_author_or_language_holdout'] is True


def test_full_artificial_recovery_seal_audit_metrics_and_exclusivity(tmp_path, monkeypatch):
    source = source_fixture()
    source.alphabet = tuple(ALPHABET)
    source.row = lambda _: source.probabilities[0]
    texts = ['abacabadabacaba'*40]
    manifest = {'old_forbidden_raw': [], 'old_forbidden_canonical': [], 'validation_source': {'artificial': True}}
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/run.EXP)
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/run.EXP)
    monkeypatch.setattr(run, 'POSITIVE', 2)
    monkeypatch.setattr(run, 'CONTROL_COUNT', 1)
    monkeypatch.setattr(run, 'CASES', 5)
    monkeypatch.setattr(run, 'SWEEPS', {'cold': 8, 'replicas': 2})
    monkeypatch.setattr(run.admitted.prior.training, 'OUT', tmp_path/'results'/'SOURCE-ACTION-TRAIN-002')
    monkeypatch.setattr(run.admitted.prior.training, 'load_prepared', lambda: (manifest, [], texts, []))

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    gold_called = []
    def save_new(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True).encode()
        if compressed:
            data = gzip.compress(data, mtime=0)
        with path.open('xb') as stream:
            stream.write(data)
        return artifact(path)

    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))

    original_metric = run.reading_endpoint_metrics
    def metric(*args):
        assert (run.OUT/'sealed-cells.json').is_file()
        gold_called.append(True)
        score = original_metric(*args)
        assert score == independent_metrics(*args)
        return score

    monkeypatch.setattr(run, 'reading_endpoint_metrics', metric)
    monkeypatch.setattr(run, 'training', SimpleNamespace(
        OUT=tmp_path/'old', old=SimpleNamespace(require_frozen=lambda *_: None), artifact=artifact, save_new=save_new,
        limit_resources=lambda *_: None, resource_report=lambda *_: {
            'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024, 'paid_spend_usd': 0}))
    monkeypatch.setattr(run.admitted.prior, 'load_archive', load)
    monkeypatch.setattr(run.admitted.prior, 'load_source', lambda: (source, {'counts': {'artificial': True}}))
    monkeypatch.setattr(run, 'require_prior', lambda: {'source_arrays': run.admitted.prior.array_identity(source)})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    outcome = run.recover('artificial-only')
    assert len(gold_called) == 5 and outcome['not_historical_or_null_semantic_classification']
    assert audit('artificial-only')['local_attempts'] == 160
    result = json.loads((run.OUT/'result.json').read_text())
    assert len(result['cells']) == 20 and result['match_is_local_attempts_not_wall_time_or_exact_CPU']
    with pytest.raises(FileExistsError):
        run.recover('artificial-only')
    with pytest.raises(FileExistsError):
        audit('artificial-only')
