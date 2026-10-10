"""Artificial full original-shape replica controller/audit; no source archive or Gold."""
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_tempered_reading_admit001 as run
from scripts.audit_tempered_reading_admit001 import audit, independent_replay
from voynich.reading_pair_rewrite import reading_actions
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.tempered_reading_trace import replica_trace


def source_fixture():
    p = np.array([[1/32]*22+[10/32]], dtype=np.float64)
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    return SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)


def initial_fixture(source):
    env = ReadingEnvironment(((0, 1, 0, 1), (0, 1)), rows=23, glyphs=6)
    state = ReadingState((-1, 0, 1)+(-1,)*20, (4, 2), ((1, 2, 1, 2), (1, 2)))
    sampler = SourceRegrowth(source, env)
    return sampler, sampler.path(forced_actions=reading_actions(env, state))


def test_alternate_replay_detects_trace_and_rng_corruption(monkeypatch):
    monkeypatch.setattr(run, 'SWEEPS', 12)
    source = source_fixture()
    sampler, initial = initial_fixture(source)
    trace = replica_trace(sampler, initial.actions, seed=96101, sweeps=12, degrees=run.DEGREES)
    result = independent_replay(source, sampler, trace)
    assert result['local_attempts'] == 48 and result['exchange_attempts'] == 18
    assert all(v['attempts'] > 0 for v in trace['summary']['by_kernel'].values())
    trace['trace'][0]['local'][0]['info']['accepted'] = not trace['trace'][0]['local'][0]['info']['accepted']
    with pytest.raises(AssertionError):
        independent_replay(source, sampler, trace)
    with pytest.raises(ValueError):
        replica_trace(sampler, initial.actions, seed=True, sweeps=12, degrees=run.DEGREES)


def test_original_shape_controller_audit_and_exclusive_namespace(tmp_path, monkeypatch):
    source = source_fixture()
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/run.EXP)
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/run.EXP)
    monkeypatch.setattr(run, 'CASES', 2)
    monkeypatch.setattr(run, 'SWEEPS', 4)

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

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

    monkeypatch.setattr(run, 'training', SimpleNamespace(
        old=SimpleNamespace(require_frozen=lambda *_: None), artifact=artifact, save_new=save_new,
        limit_resources=lambda *_: None, resource_report=lambda *_: {
            'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024, 'paid_spend_usd': 0}))
    monkeypatch.setattr(run.prior, 'load_archive', load)
    monkeypatch.setattr(run.prior, 'load_source', lambda: (source, None))
    sampler, initial = initial_fixture(source)
    frames = []
    for case in range(run.CASES):
        for phase in run.PHASES:
            spec = save_new(tmp_path/f'inputs/{case}-{phase}.gz', {'records': list(map(list, sampler.env.records)),
                'base': run.prior.prior.reading_receipt(initial)}, True)
            frames.append({'case': case, 'kind': 'artificial', 'phase': phase, 'archive': spec})
    monkeypatch.setattr(run, 'require_prior', lambda: {'cells': frames, 'source_arrays': run.prior.array_identity(source)})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    assert all(run.admission('artificial-only').values())
    assert audit('artificial-only')['local_attempts'] == 64
    result = json.loads((run.OUT/'result.json').read_text())
    assert len(result['cells']) == 4 and result['no_gold_metrics_neural_training_or_optimizer']
    assert result['no_recovery_mixing_or_historical_claim']
    with pytest.raises(FileExistsError):
        run.admission('artificial-only')
    with pytest.raises(FileExistsError):
        audit('artificial-only')
